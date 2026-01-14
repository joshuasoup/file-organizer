from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fileorg.chat.tools.common import ToolResult, to_json
from fileorg.config import AppConfig
from fileorg.indexer.scan import (
    DEFAULT_ORGANIZED_SCORE_THRESHOLD,
    assess_folders_from_paths,
    should_ignore,
)
from fileorg.store import ChromaStore

_STOP_TOKENS = {
    "file",
    "files",
    "copy",
    "final",
    "notes",
    "doc",
    "docs",
    "document",
    "documents",
    "page",
    "pages",
    "midterm",
    "exam",
    "test",
}


def _is_clean_dir_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name))


def _sanitize_component(raw: str, fallback: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", raw.lower())
    if not tokens:
        tokens = [fallback]
    segment = "-".join(tokens).strip("-")
    segment = segment[:60] or fallback
    return segment


def _extract_topic_slug(basenames: list[str], fallback: str) -> str:
    course_match = None
    course_pattern = re.compile(r"[a-z]{3,5}[0-9]{3,4}[a-z]?")
    for name in basenames:
        found = course_pattern.findall(name.lower())
        if found:
            course_match = found[0]
            break

    counts: dict[str, int] = {}
    for name in basenames:
        for tok in re.findall(r"[a-z0-9]+", name.lower()):
            if len(tok) < 3:
                continue
            if tok.isdigit():
                continue
            if tok in _STOP_TOKENS:
                continue
            counts[tok] = counts.get(tok, 0) + 1

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tokens = [t for t, c in ordered if c >= 2][:3] or [t for t, _ in ordered[:3]]

    parts: list[str] = []
    if course_match:
        parts.append(course_match)
    parts.extend(top_tokens)
    slug_raw = "-".join(parts) if parts else fallback
    return _sanitize_component(slug_raw, fallback)


def _normalize_cluster_folder(raw_name: str, idx: int, topic_slug: str) -> str:
    label = _sanitize_component(raw_name, topic_slug or f"group-{idx:03d}")
    if label.startswith("collection-") or label.startswith("group-"):
        return label.replace("collection-", "collection_").replace("group-", "group_")
    return label


def _normalize_path(
    raw_path: str,
    default_leaf: str,
    idx: int,
    basenames: list[str],
    topic_slug: str,
) -> str:
    pieces = [p for p in (raw_path or "").replace("\\", "/").split("/") if p]
    root_candidate = pieces[0].lower() if pieces else ""
    root = _sanitize_component(root_candidate or topic_slug or "root", topic_slug or "root")
    remainder = pieces[1:] if root_candidate else pieces
    normalized_parts = [root]

    topic = _sanitize_component(topic_slug or "collection", "collection")
    if not remainder:
        remainder = [topic, default_leaf]
    elif len(remainder) == 1:
        remainder = [topic] + remainder

    for part in remainder[:-1]:
        normalized_parts.append(_sanitize_component(part, "collection"))

    leaf_raw = remainder[-1] if remainder else default_leaf
    normalized_parts.append(_normalize_cluster_folder(leaf_raw, idx, topic))
    return "/".join(normalized_parts)


def _within_depth(path: str, root: Path | None, max_depth: int) -> bool:
    if max_depth < 0:
        return True
    candidate = Path(path)
    rel = candidate
    if root is not None:
        try:
            rel = candidate.relative_to(root)
        except ValueError:
            rel = candidate
    dir_depth = max(0, len(rel.parts) - 1)
    return dir_depth <= max_depth


def _list_existing_dirs(
    root: Path,
    config: AppConfig | None,
    max_depth: int = 2,
    max_dirs: int = 400,
) -> list[Path]:
    existing: list[Path] = []
    for current_root, dirs, _ in os.walk(root):
        rel_parts = Path(current_root).relative_to(root).parts
        depth = len(rel_parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        pruned: list[str] = []
        for name in dirs:
            candidate = Path(current_root) / name
            if name.startswith("."):
                continue
            if config is not None and should_ignore(candidate, root, config):
                continue
            pruned.append(name)
            if len(existing) < max_dirs:
                existing.append(candidate)
        dirs[:] = pruned
        if len(existing) >= max_dirs:
            break
    return existing


def _choose_kmeans_k(total: int) -> int:
    if total <= 15:
        return 1
    rough = int(math.sqrt(total)) + 1
    return max(2, min(12, rough))


def _kmeans_partition(vectors: np.ndarray, k: int, max_iter: int = 25) -> np.ndarray:
    if len(vectors) == 0:
        return np.array([], dtype=int)
    k = max(1, min(k, len(vectors)))
    if k == 1:
        return np.zeros(len(vectors), dtype=int)

    rng = np.random.default_rng(42)
    centroids = vectors[rng.choice(len(vectors), size=k, replace=False)]
    labels = np.zeros(len(vectors), dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = vectors[labels == idx]
            if len(members) == 0:
                centroids[idx] = vectors[rng.integers(0, len(vectors))]
            else:
                centroids[idx] = members.mean(axis=0)
    return labels


def _compute_centroid(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.array([])
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm
    return centroid


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) or 1e-9
    return float(np.dot(vec_a, vec_b) / denom)


def _basename_token_similarity(a: list[str], b: list[str]) -> float:
    def tokens(names: list[str]) -> set[str]:
        toks: set[str] = set()
        for name in names:
            for tok in re.findall(r"[a-z0-9]+", name.lower()):
                if len(tok) < 3:
                    continue
                if tok.isdigit():
                    continue
                if tok in _STOP_TOKENS:
                    continue
                toks.add(tok)
        return toks

    a_tokens = tokens(a)
    b_tokens = tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return float(len(overlap) / len(union)) if union else 0.0


def _cluster_confidence(vectors: np.ndarray, probabilities: list[float] | None) -> float:
    if len(vectors) == 0:
        return 0.0
    centroid = _compute_centroid(vectors)
    distances = np.linalg.norm(vectors - centroid, axis=1)
    cohesion = 1.0 / (1.0 + float(np.mean(distances)))
    size_score = min(1.0, math.log1p(len(vectors)) / math.log(50))
    prob_score = float(np.mean(probabilities)) if probabilities is not None and len(probabilities) > 0 else 0.5
    confidence = 0.45 * prob_score + 0.35 * cohesion + 0.2 * size_score
    return round(min(1.0, max(0.0, confidence)), 3)


def _merge_similar_clusters(clusters: list[dict], similarity_threshold: float = 0.92) -> list[dict]:
    merged: list[dict] = []
    for cluster in clusters:
        centroid = cluster.get("centroid", np.array([]))
        if centroid.size == 0:
            merged.append(cluster)
            continue
        merged_into = None
        for existing in merged:
            existing_centroid = existing.get("centroid", np.array([]))
            if existing_centroid.size == 0:
                continue
            existing_vectors = existing.get("vectors", np.array([]))
            vectors = cluster.get("vectors", np.array([]))
            if existing_vectors.size == 0 or vectors.size == 0:
                continue
            if existing_vectors.shape[1] != vectors.shape[1]:
                continue
            cosine_sim = _cosine_similarity(centroid, existing_centroid)
            name_sim = _basename_token_similarity(
                existing.get("basenames", []), cluster.get("basenames", [])
            )
            if cosine_sim >= similarity_threshold:
                existing["paths"].extend(cluster.get("paths", []))
                existing["basenames"].extend(cluster.get("basenames", []))
                existing["vectors"] = np.vstack([existing_vectors, vectors])
                existing["probabilities"].extend(cluster.get("probabilities", []))
                existing["centroid"] = _compute_centroid(existing["vectors"])
                existing["method"] = f"{existing.get('method', 'cluster')},merged"
                merged_into = existing
                break
        if merged_into is None:
            merged.append(cluster)
    return merged


def _best_existing_dir(
    topic_slug: str,
    basenames: list[str],
    existing_dirs: list[Path],
    scan_root: Path | None,
    threshold: float = 0.25,
) -> tuple[str | None, float]:
    best_dir: Path | None = None
    best_score = threshold
    for dir_path in existing_dirs:
        dir_name = dir_path.name
        if not _is_clean_dir_name(dir_name):
            continue
        file_sim = _basename_token_similarity(basenames, [dir_name])
        slug_sim = _basename_token_similarity([topic_slug], [dir_name])
        score = 0.6 * file_sim + 0.4 * slug_sim
        if score > best_score:
            best_score = score
            best_dir = dir_path
    if best_dir is None:
        return None, 0.0
    rel = best_dir
    if scan_root is not None:
        try:
            rel = best_dir.relative_to(scan_root)
        except ValueError:
            rel = best_dir
    return rel.as_posix(), round(best_score, 3)


def _sample_content_snippets(
    chroma: ChromaStore, paths: list[str], max_files: int = 3, max_chars: int = 320
) -> list[str]:
    snippets: list[str] = []
    text_store = getattr(chroma, "_text", None)
    if text_store is None:
        return snippets
    for path in paths[:max_files]:
        try:
            data = text_store.get(where={"path": path}, include=["documents"], limit=1)
        except Exception:
            continue
        docs = data.get("documents") if isinstance(data, dict) else None
        if not docs:
            continue
        doc_entry = docs[0]
        if isinstance(doc_entry, list):
            doc_entry = doc_entry[0] if doc_entry else ""
        if not doc_entry:
            continue
        snippet = str(doc_entry).replace("\n", " ").strip()
        if snippet:
            snippets.append(snippet[:max_chars])
    return snippets


def tool_suggest_structure(
    chroma: ChromaStore,
    client,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    scan_root: Path | None = None,
    config: AppConfig | None = None,
    max_depth: int = 1,
    organized_threshold: float = DEFAULT_ORGANIZED_SCORE_THRESHOLD,
    usage_callback: Callable[[Any], None] | None = None,
) -> ToolResult:
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return ToolResult(
            "suggest_structure",
            "hdbscan not installed; cannot cluster. Install dependencies and re-run.",
        )

    try:
        embeddings = chroma.file_embeddings()
        filtered_by_depth = scan_root is not None
        if scan_root is not None:
            filtered_embeddings = {
                path: vec
                for path, vec in embeddings.items()
                if _within_depth(path, scan_root, max_depth)
            }
            embeddings = filtered_embeddings

        existing_dirs: list[Path] = []
        if scan_root is not None:
            try:
                existing_dirs = _list_existing_dirs(
                    scan_root, config, max_depth=2, max_dirs=600
                )
            except Exception:
                existing_dirs = []

        filtered_by_org = False
        _, organized_dirs = assess_folders_from_paths(
            list(embeddings.keys()),
            embeddings=embeddings,
            threshold=organized_threshold,
        )
        if organized_dirs:
            organized_sorted = sorted(organized_dirs, key=lambda p: len(p.parts))
            embeddings = {
                path: vec
                for path, vec in embeddings.items()
                if not any(Path(path).is_relative_to(org) for org in organized_sorted)
            }
            filtered_by_org = True
        if not embeddings:
            message = "No embeddings available. Run indexing first."
            if filtered_by_depth:
                message = "No embeddings available within the scan depth limit. Run indexing first."
            if filtered_by_org:
                message = "No embeddings available after skipping folders already organized."
            return ToolResult(
                "suggest_structure",
                message,
            )

        max_files_to_process = 5000
        total_files = len(embeddings)
        if total_files > max_files_to_process:
            import random
            sample_paths = random.sample(list(embeddings.keys()), max_files_to_process)
            embeddings = {path: embeddings[path] for path in sample_paths}

        lengths: dict[int, list[str]] = defaultdict(list)
        for path, vec in embeddings.items():
            lengths[len(vec)].append(path)
        if not lengths:
            return ToolResult(
                "suggest_structure",
                "No usable embeddings found.",
            )

        raw_clusters: list[dict] = []
        for dim, paths in lengths.items():
            if not paths:
                continue
            vectors = np.stack([embeddings[p] for p in paths])
            kmeans_k = _choose_kmeans_k(len(paths))
            tier1_labels = _kmeans_partition(vectors, kmeans_k)
            num_coarse = int(tier1_labels.max()) + 1 if tier1_labels.size else 1

            for coarse_label in range(num_coarse):
                member_indices = np.where(tier1_labels == coarse_label)[0]
                if len(member_indices) == 0:
                    continue
                member_paths = [paths[i] for i in member_indices]
                member_vecs = vectors[member_indices]
                clusters_here: list[dict] = []
                derived_min_cluster_size = max(15, len(member_paths) // 15)
                can_hdbscan = len(member_paths) >= derived_min_cluster_size

                if can_hdbscan:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=derived_min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.3,
                    )
                    labels = clusterer.fit_predict(member_vecs)
                    probs_array = getattr(clusterer, "probabilities_", None)
                    label_set = {int(l) for l in labels if l != -1}
                    for label in label_set:
                        idxs = [i for i, l in enumerate(labels) if l == label]
                        if not idxs:
                            continue
                        c_paths = [member_paths[i] for i in idxs]
                        c_vecs = member_vecs[idxs]
                        c_probs = [float(probs_array[i]) for i in idxs] if probs_array is not None else []
                        clusters_here.append(
                            {
                                "paths": c_paths,
                                "basenames": [Path(p).name for p in c_paths],
                                "vectors": c_vecs,
                                "probabilities": c_probs,
                                "method": "kmeans+hdbscan",
                                "centroid": _compute_centroid(c_vecs),
                            }
                        )

                    noise_indices = [i for i, l in enumerate(labels) if l == -1]
                    if noise_indices:
                        noise_paths = [member_paths[i] for i in noise_indices]
                        noise_vecs = member_vecs[noise_indices]
                        clusters_here.append(
                            {
                                "paths": noise_paths,
                                "basenames": [Path(p).name for p in noise_paths],
                                "vectors": noise_vecs,
                                "probabilities": [float(probs_array[i]) for i in noise_indices]
                                if probs_array is not None
                                else [],
                                "method": "kmeans_noise",
                                "centroid": _compute_centroid(noise_vecs),
                            }
                        )

                if not clusters_here:
                    clusters_here.append(
                        {
                            "paths": member_paths,
                            "basenames": [Path(p).name for p in member_paths],
                            "vectors": member_vecs,
                            "probabilities": [],
                            "method": "kmeans_only",
                            "centroid": _compute_centroid(member_vecs),
                        }
                    )

                raw_clusters.extend(clusters_here)

        merged_clusters = _merge_similar_clusters(raw_clusters, similarity_threshold=0.92)
        if len(merged_clusters) > 40:
            merged_clusters = _merge_similar_clusters(merged_clusters, similarity_threshold=0.85)
        if not merged_clusters:
            message = "No meaningful clusters found. Try adjusting min_cluster_size or min_samples parameters."
            if total_files > max_files_to_process:
                message += f" (Analyzed {max_files_to_process} files sampled from {total_files} total)"
            return ToolResult("suggest_structure", message)

        suggestions = []
        cluster_idx = 0
        max_clusters_to_name = 30
        clusters_processed = 0
        sorted_clusters = sorted(merged_clusters, key=lambda c: len(c.get("paths", [])), reverse=True)

        for cluster in sorted_clusters:
            basenames = cluster.get("basenames", [])
            topic_slug = _extract_topic_slug(basenames, f"group-{cluster_idx:03d}")
            normalized_name = _normalize_cluster_folder(topic_slug, cluster_idx, topic_slug)
            sorted_paths = sorted(cluster.get("paths", []))
            confidence = _cluster_confidence(cluster.get("vectors", np.array([])), cluster.get("probabilities"))

            existing_path = None
            existing_score = 0.0
            if existing_dirs:
                existing_path, existing_score = _best_existing_dir(
                    topic_slug, basenames, existing_dirs, scan_root
                )
            if existing_path:
                suggestions.append(
                    {
                        "folder": Path(existing_path).name or normalized_name,
                        "path": existing_path,
                        "count": len(sorted_paths),
                        "sample_files": basenames[:5],
                        "paths": sorted_paths,
                        "confidence": confidence,
                        "strategy": f"{cluster.get('method', 'cluster')},existing_match",
                        "existing_folder": True,
                        "existing_score": existing_score,
                    }
                )
                cluster_idx += 1
                clusters_processed += 1
                continue

            if clusters_processed >= max_clusters_to_name:
                suggestions.append(
                    {
                        "folder": normalized_name,
                        "path": _normalize_path("", normalized_name, cluster_idx, basenames, topic_slug),
                        "count": len(sorted_paths),
                        "sample_files": basenames[:5],
                        "paths": sorted_paths,
                        "confidence": confidence,
                        "strategy": cluster.get("method", "cluster"),
                    }
                )
                cluster_idx += 1
                continue

            sample = ", ".join(basenames[:10])
            snippets = _sample_content_snippets(chroma, sorted_paths)
            prompt = (
                "Using this naming guide, output ONE relative folder path (no filename) for these related files.\n"
                "- Choose a root that fits the hierarchy (you pick the wording): work/professional, personal/life, learning/academy, creative/media, system/dev, random/junk.\n"
                "- Lowercase only. Use underscores between sections, hyphens inside sections. No spaces or special characters.\n"
                "- Prefer 2-3 levels deep (e.g., coursework/comp2404/course-materials).\n"
                "- Keep segments concise (<= 60 chars). No trailing slash. Output only the path.\n"
                f"Sample files: {sample}"
            )
            if snippets:
                snippet_block = "\n".join(f"- {s}" for s in snippets)
                prompt += f"\nContent clues (truncated):\n{snippet_block}"
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.2,
                )
                if usage_callback:
                    usage_callback(resp)
                name = (resp.choices[0].message.content or "").strip().splitlines()[0]
            except Exception:
                name = ""
            topic_slug = _extract_topic_slug(basenames, f"group-{cluster_idx:03d}")
            normalized_name = _normalize_cluster_folder(name or topic_slug, cluster_idx, topic_slug)
            suggested_path = _normalize_path(name, normalized_name, cluster_idx, basenames, topic_slug)
            suggestions.append(
                {
                    "folder": normalized_name,
                    "path": suggested_path,
                    "count": len(sorted_paths),
                    "sample_files": basenames[:5],
                    "paths": sorted_paths,
                    "confidence": confidence,
                    "strategy": cluster.get("method", "cluster"),
                }
            )
            cluster_idx += 1
            clusters_processed += 1

        if not suggestions:
            message = "No meaningful clusters found. Try adjusting min_cluster_size or min_samples parameters."
            if total_files > max_files_to_process:
                message += f" (Analyzed {max_files_to_process} files sampled from {total_files} total)"
            return ToolResult("suggest_structure", message)

        return ToolResult("suggest_structure", to_json(suggestions))
    except Exception as e:
        error_msg = f"Error during structure analysis: {str(e)}"
        return ToolResult("suggest_structure", error_msg)
