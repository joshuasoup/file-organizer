from __future__ import annotations

import math, os, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fileorg.chat.tools.common import ToolResult, to_json
from fileorg.config import AppConfig
from fileorg.indexer.scan import DEFAULT_ORGANIZED_SCORE_THRESHOLD, assess_folders_from_paths, should_ignore
from fileorg.store import ChromaStore

# --- UTILS & VECTOR MATH ---

def _clean_filename(filename: str) -> str:
    """Removes extensions and common versioning noise for better similarity checks."""
    name = Path(filename).stem.lower()
    name = re.sub(r"(_v\d+|copy|final|draft|scan|document)", "", name)
    return re.sub(r"[^a-z0-9]", " ", name).strip()


def _coerce_embedding_vector(vec: Any) -> np.ndarray | None:
    """Best-effort conversion of raw embedding to a flat numeric vector."""
    try:
        arr = np.asarray(vec, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    return arr.reshape(-1)


def _sanitize_segment(raw: str, fallback: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    if not tokens:
        return fallback
    segment = "-".join(tokens).strip("-")
    return segment[:60] or fallback


def _normalize_suggested_path(raw: str, default: str = "unsorted") -> str:
    raw = (raw or "").lower().replace("\\", "/")
    parts = []
    for piece in raw.split("/"):
        cleaned = piece.strip()
        cleaned = cleaned.replace("context:", "").replace("project:", "")
        if cleaned in {"context", "project", "folder", "path"}:
            cleaned = ""
        sanitized = _sanitize_segment(cleaned, "")
        if sanitized:
            parts.append(sanitized)
        if len(parts) == 2:
            break
    if not parts:
        return default
    return "/".join(parts)


def _clickable(path: str) -> str:
    """Escapes spaces so IDE/terminal command-click works reliably."""
    return path.replace(" ", "\\ ")


def _prefix_root(path_str: str, prefix: str = "_") -> str:
    """Adds a prefix to the first path segment (preserves leading '*' for uncertainty)."""
    if not prefix:
        return path_str
    star = path_str.startswith("*")
    core = path_str[1:] if star else path_str
    parts = core.split("/")
    if not parts or parts[0].startswith(prefix):
        return path_str
    parts[0] = f"{prefix}{parts[0]}"
    new_core = "/".join(parts)
    return f"*{new_core}" if star else new_core


def _apply_root_prefix(suggestions: list[dict], prefix: str = "_") -> list[dict]:
    if not prefix:
        return suggestions
    for s in suggestions:
        path = str(s.get("path", ""))
        prefixed = _prefix_root(path, prefix)
        s["path"] = prefixed
        s["folder"] = Path(prefixed).name
        s["path_clickable"] = _clickable(prefixed)
    return suggestions


def _derive_name_from_paths(paths: list[str], max_tokens: int = 2, min_count: int = 2) -> str | None:
    """Builds a backup name from common filename tokens."""
    token_counts: dict[str, int] = {}
    for p in paths:
        stem = Path(p).stem.lower()
        for tok in re.findall(r"[a-z0-9]+", stem):
            if tok.isdigit():
                continue
            token_counts[tok] = token_counts.get(tok, 0) + 1
    common = [t for t, c in sorted(token_counts.items(), key=lambda x: (-x[1], x[0])) if c >= min_count]
    if not common:
        return None
    name = "-".join(common[:max_tokens])[:60].strip("-")
    return name or None


def _llm_bucket_from_files(
    paths: list[str],
    client,
    medoids: list[str] | None = None,
    outliers: list[str] | None = None,
) -> str | None:
    """Asks the LLM for a concise bucket name based on filenames."""
    if client is None:
        return None
    files_sample = [Path(p).name for p in paths[:10]]
    core = ", ".join(medoids[:5]) if medoids else ", ".join(files_sample[:5])
    outs = ", ".join(outliers[:3]) if outliers else ""
    prompt = (
        "You are naming a folder for similar files. Return a short lowercase hyphenated name (no prefixes like '*'). "
        "Avoid 'unsorted'.\n"
        f"Example files: {core or 'n/a'}\n"
        f"Outliers: {outs or 'n/a'}\n"
        "Return only the name, no slashes."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.2,
        )
        name = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        return None
    name = name.replace("/", " ").replace("*", "")
    return _sanitize_segment(name, "")


def _get_fused_embedding(content_vec: np.ndarray, filename: str, weight_content: float = 0.7) -> np.ndarray:
    """Combines content embedding with a lightweight filename feature vector."""
    if content_vec.size == 0:
        return content_vec
    c_norm = content_vec / (np.linalg.norm(content_vec) or 1e-9)
    fn_cleaned = _clean_filename(filename)
    fn_hash = np.zeros(content_vec.shape[0])
    for i, char in enumerate(fn_cleaned[:50]):
        fn_hash[i % content_vec.shape[0]] += ord(char)
    fn_norm = fn_hash / (np.linalg.norm(fn_hash) or 1e-9)
    return (weight_content * c_norm) + ((1 - weight_content) * fn_norm)

def _compute_centroid(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.array([])
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else centroid


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _analyze_cluster_metrics(vectors: np.ndarray) -> dict:
    """Consolidated math pass for cohesion, diversity, and specificity."""
    if len(vectors) == 0:
        return {"cohesion": 0, "diversity": 0, "label": "low", "spec": "general"}
    centroid = _compute_centroid(vectors)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    cohesion = float(np.mean([_cosine_sim(v, centroid) for v in vectors]))
    diversity = float(np.std(distances))
    if cohesion >= 0.82 and diversity < 0.2:
        label, spec = "high", "specific"
    elif cohesion >= 0.65:
        label, spec = "medium", "balanced"
    else:
        label, spec = "low", "general"
    return {"cohesion": round(cohesion, 3), "diversity": round(diversity, 3), "label": label, "spec": spec}


def _get_medoids(vectors: np.ndarray, paths: list[str], n: int = 5) -> list[str]:
    if len(vectors) == 0 or not paths or n == 0:
        return []
    centroid = vectors.mean(axis=0)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    take = max(1, min(abs(n), len(paths)))
    medoid_idx = np.argsort(distances)
    if n < 0:
        medoid_idx = medoid_idx[::-1]
    medoid_idx = medoid_idx[:take]
    return [Path(paths[i]).name for i in medoid_idx]

def _is_leaky(
    vectors: np.ndarray, std_threshold: float = 0.45, cohesion_threshold: float = 0.55
) -> bool:
    """Checks if the cluster spreads too widely (std dev) or is low cohesion."""
    if len(vectors) < 4:
        return False
    centroid = _compute_centroid(vectors)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    distance_std = float(np.std(distances))
    cohesion = float(np.mean([_cosine_sim(v, centroid) for v in vectors]))
    return (distance_std > std_threshold) or (cohesion < cohesion_threshold)


def _refine_clusters(
    raw_clusters: list[dict],
    min_size: int = 2,
    std_threshold: float = 0.45,
    cohesion_threshold: float = 0.55,
) -> list[dict]:
    """Recursively splits 'leaky' clusters to prevent cross-contamination."""
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return raw_clusters

    refined = []
    for cluster in raw_clusters:
        vecs = cluster["vectors"]
        if len(vecs) < 10 or not _is_leaky(vecs, std_threshold, cohesion_threshold):
            refined.append(cluster)
            continue

        sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, cluster_selection_epsilon=0.2)
        sub_labels = sub_clusterer.fit_predict(vecs)

        if len(set(sub_labels)) <= 1:
            refined.append(cluster)
        else:
            for sl in set(sub_labels):
                if sl == -1:
                    continue
                idx = np.where(sub_labels == sl)[0]
                if len(idx) == 0:
                    continue
                sub_vecs = vecs[idx]
                refined.append(
                    {
                        "paths": [cluster["paths"][i] for i in idx],
                        "vectors": sub_vecs,
                        "centroid": _compute_centroid(sub_vecs),
                    }
                )
    return refined

def _collect_unembedded_paths(
    scan_root: Path,
    fused_paths: set[str],
    organized: set[Path],
    max_depth: int,
    config: AppConfig | None = None,
) -> list[str]:
    """Walks the tree to gather files that were not embedded (e.g., zips, dmg, media)."""
    cfg = config or AppConfig()
    follow_symlinks = cfg.scan.follow_symlinks
    extras: list[str] = []
    for current_root, dirs, files in os.walk(scan_root, followlinks=follow_symlinks):
        current_path = Path(current_root)
        pruned_dirs = []
        for name in dirs:
            candidate = current_path / name
            if not follow_symlinks and candidate.is_symlink():
                continue
            if should_ignore(candidate, scan_root, cfg):
                continue
            pruned_dirs.append(name)
        dirs[:] = pruned_dirs

        for name in files:
            path = current_path / name
            if not follow_symlinks and path.is_symlink():
                continue
            if should_ignore(path, scan_root, cfg):
                continue
            try:
                rel_parts = path.relative_to(scan_root).parts
            except Exception:
                continue
            if len(rel_parts) > max_depth:
                continue
            if str(path) in fused_paths:
                continue
            if organized and any(path.is_relative_to(org) for org in organized):
                continue
            extras.append(str(path))
    return extras


def _attach_unembedded_files(
    suggestions: list[dict],
    unembedded_paths: list[str],
    fallback_path: str = "unsorted",
) -> list[dict]:
    if not unembedded_paths:
        return suggestions

    base = fallback_path.rstrip("/")

    def _ext_bucket(path_str: str) -> str:
        p = Path(path_str)
        ext = (p.suffixes[-1].lstrip(".") if p.suffixes else "") or "noext"
        sanitized_ext = _sanitize_segment(ext, "noext")
        return f"{base}/{sanitized_ext}"

    def _ensure_bucket(bucket_path: str) -> int:
        for idx, s in enumerate(suggestions):
            if str(s.get("path", "")) == bucket_path:
                return idx
        suggestions.append(
            {
                "folder": Path(bucket_path).name,
                "path": bucket_path,
                "path_clickable": _clickable(bucket_path),
                "count": 0,
                "sample_files": [],
                "paths": [],
                "paths_clickable": [],
                "metrics": {"label": "low", "spec": "general"},
            }
        )
        return len(suggestions) - 1

    for extra in unembedded_paths:
        bucket_path = _ext_bucket(extra)
        target_idx = _ensure_bucket(bucket_path)
        suggestions[target_idx].setdefault("paths", []).append(extra)
        suggestions[target_idx].setdefault("paths_clickable", []).append(_clickable(extra))

    for s in suggestions:
        s["count"] = len(s.get("paths", []))
    return suggestions

# --- CLUSTERING FLOW ---

def _merge_clusters(clusters: list[dict], threshold: float = 0.94) -> list[dict]:
    merged = []
    for c in clusters:
        for m in merged:
            if _cosine_sim(c["centroid"], m["centroid"]) >= threshold:
                m["paths"].extend(c["paths"])
                m["vectors"] = np.vstack([m["vectors"], c["vectors"]])
                m["centroid"] = _compute_centroid(m["vectors"])
                break
        else: merged.append(c)
    return merged


def _reassign_leaks(clusters: list[dict], margin: float = 0.05) -> list[dict]:
    """Moves points to a different cluster if its centroid is clearly closer."""
    if len(clusters) < 2:
        return clusters

    centroids = [c["centroid"] for c in clusters]
    reassigned: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for idx, cluster in enumerate(clusters):
        for path, vec in zip(cluster["paths"], cluster["vectors"]):
            sims = [ _cosine_sim(vec, c) for c in centroids ]
            best_idx = int(np.argmax(sims))
            current_sim = sims[idx]
            best_sim = sims[best_idx]
            if best_idx != idx and (best_sim - current_sim) >= margin:
                target = best_idx
            else:
                target = idx
            reassigned[target].append((path, vec))

    repaired = []
    for _, items in sorted(reassigned.items()):
        if not items:
            continue
        paths, vecs = zip(*items)
        vec_array = np.stack(vecs)
        repaired.append(
            {
                "paths": list(paths),
                "vectors": vec_array,
                "centroid": _compute_centroid(vec_array),
            }
        )
    return repaired

# --- MAIN TOOL ---

def tool_suggest_structure(chroma: ChromaStore, client, scan_root: Path | None = None, **kwargs) -> ToolResult:
    try: import hdbscan
    except ImportError: return ToolResult("suggest_structure", "hdbscan not installed.")

    config: AppConfig | None = kwargs.get("config")
    max_depth = kwargs.get("max_depth", 2)
    leak_std_threshold = kwargs.get("leak_std_threshold", 0.45)
    leak_cohesion_threshold = kwargs.get("leak_cohesion_threshold", 0.55)
    root_prefix = kwargs.get("root_prefix", "_")

    raw_embeddings = chroma.file_embeddings()
    fused_map: dict[str, np.ndarray] = {}
    for path, vec in raw_embeddings.items():
        coerced = _coerce_embedding_vector(vec)
        if coerced is None or coerced.size == 0:
            continue
        fused_map[path] = _get_fused_embedding(coerced, Path(path).name)

    # 2. Filter out already organized or out-of-depth files
    if scan_root:
        pruned: dict[str, np.ndarray] = {}
        for p, v in fused_map.items():
            try:
                rel_parts = Path(p).relative_to(scan_root).parts
            except Exception:
                continue
            if len(rel_parts) <= max_depth:
                pruned[p] = v
        fused_map = pruned

    _, organized = assess_folders_from_paths(list(fused_map.keys()), embeddings=fused_map)
    fused_map = {p: v for p, v in fused_map.items() if not any(Path(p).is_relative_to(org) for org in organized)}
    
    unembedded_paths: list[str] = []
    if scan_root:
        unembedded_paths = _collect_unembedded_paths(
            scan_root=scan_root,
            fused_paths=set(fused_map.keys()),
            organized=organized,
            max_depth=max_depth,
            config=config,
        )

    if not fused_map and not unembedded_paths:
        return ToolResult("suggest_structure", "No new files to organize.")

    # 3. HDBSCAN Clustering on Fused Vectors (per-dimension to avoid shape issues)
    groups: dict[int, list[str]] = defaultdict(list)
    for path, vec in fused_map.items():
        groups[vec.shape[0]].append(path)

    suggestions = []
    max_clusters = 30
    for dim, paths in groups.items():
        if not paths:
            continue
        vectors = np.stack([fused_map[p] for p in paths])
        if vectors.size == 0:
            continue
        if len(paths) < 2:
            labels = np.zeros(len(paths), dtype=int)
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=kwargs.get('min_size', 4),
                cluster_selection_epsilon=0.35,
            )
            try:
                labels = clusterer.fit_predict(vectors)
            except ValueError:
                labels = np.zeros(len(paths), dtype=int)

        raw_clusters = []
        for l in set(labels):
            if l == -1:
                continue
            idx = np.where(labels == l)[0]
            if len(idx) == 0:
                continue
            c_vecs = vectors[idx]
            raw_clusters.append(
                {
                    "paths": [paths[i] for i in idx],
                    "vectors": c_vecs,
                    "centroid": _compute_centroid(c_vecs),
                }
            )
        noise_idx = np.where(labels == -1)[0] if len(set(labels)) > 1 else np.array([], dtype=int)
        if len(raw_clusters) == 0 and len(paths) > 0:
            raw_clusters.append(
                {
                    "paths": list(paths),
                    "vectors": vectors,
                    "centroid": _compute_centroid(vectors),
                }
            )
        elif len(noise_idx) > 0:
            noise_vecs = vectors[noise_idx]
            raw_clusters.append(
                {
                    "paths": [paths[i] for i in noise_idx],
                    "vectors": noise_vecs,
                    "centroid": _compute_centroid(noise_vecs),
                }
            )

        merged = _merge_clusters(raw_clusters)
        refined = _refine_clusters(
            merged,
            min_size=kwargs.get("refine_min_size", 2),
            std_threshold=leak_std_threshold,
            cohesion_threshold=leak_cohesion_threshold,
        )
        stabilized = _reassign_leaks(refined, margin=kwargs.get("reassign_margin", 0.06))

        for i, c in enumerate(sorted(stabilized, key=lambda x: len(x["paths"]), reverse=True)):
            if len(suggestions) >= max_clusters:
                break
            metrics = _analyze_cluster_metrics(c["vectors"])
            medoids = _get_medoids(c["vectors"], c["paths"])
            outliers = _get_medoids(c["vectors"], c["paths"], n=-5)
            is_leaky_cluster = _is_leaky(
                c["vectors"],
                std_threshold=leak_std_threshold,
                cohesion_threshold=leak_cohesion_threshold,
            )
            cluster_paths = sorted(c["paths"])
            prompt = (
                "Role: Expert File Organizer. Output format: 'context/project'.\n"
                f"Cluster Density: {metrics['label']} (Cohesion: {metrics['cohesion']}, Diversity: {metrics['diversity']}).\n"
                f"Confidence Level: {metrics['spec']}. Potential leakage: {'yes' if is_leaky_cluster else 'no'}.\n"
                f"CORE files: {', '.join(medoids) if medoids else 'n/a'}\n"
                f"OUTLIER files: {', '.join(outliers) if outliers else 'n/a'}\n\n"
                "If the OUTLIERS look off-topic vs the CORE files, treat the cluster as leaky. Still propose your best "
                "context/project guess (or a filename-derived bucket) but prefix it with '*' to signal uncertainty. If you truly can't place it, use '*unsorted'.\n"
                "- Context must be a real-world area (company, team, domain like marketing, psychology, backend), not the literal word 'context'.\n"
                "- Project must be a clean, specific topic name without prefixes like 'project:'.\n"
                "- Use lowercase, hyphens inside segments, and '/' between context and project.\n"
                "- If confidence is 'general' or the cluster feels mixed, prefix the path with '*' to signal uncertainty. Otherwise, be specific but concise."
            )

            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=25,
                    temperature=0.1,
                )
                suggested_path = (resp.choices[0].message.content or "").strip()
            except Exception:
                suggested_path = "unsorted"

            low_confidence = metrics["spec"] == "general" and metrics["cohesion"] < leak_cohesion_threshold
            suggested_path = _normalize_suggested_path(suggested_path, default="unsorted")
            if suggested_path.lstrip("*") == "unsorted" and len(cluster_paths) >= 2:
                candidate = _llm_bucket_from_files(cluster_paths, client, medoids, outliers)
                if not candidate:
                    candidate = _derive_name_from_paths(cluster_paths)
                if candidate:
                    suggested_path = _normalize_suggested_path(candidate, default="unsorted")
            if low_confidence or is_leaky_cluster:
                if not suggested_path.startswith("*"):
                    suggested_path = f"*{suggested_path}"

            suggestions.append(
                {
                    "folder": Path(suggested_path).name,
                    "path": suggested_path,
                    "path_clickable": _clickable(suggested_path),
                    "count": len(c["paths"]),
                    "sample_files": medoids,
                    "paths": cluster_paths,
                    "paths_clickable": [_clickable(p) for p in cluster_paths],
                    "metrics": metrics,
                    "dim": dim,
                }
            )

    if not suggestions:
        suggestions = _attach_unembedded_files([], unembedded_paths)
        if not suggestions:
            return ToolResult("suggest_structure", "No meaningful clusters found.")
    else:
        suggestions = _attach_unembedded_files(suggestions, unembedded_paths)

    suggestions = _apply_root_prefix(suggestions, prefix=root_prefix)

    return ToolResult("suggest_structure", to_json(suggestions))
