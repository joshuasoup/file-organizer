from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from fileorg.config import AppConfig
from fileorg.indexer.types import FileInfo, FileType

DEFAULT_ORGANIZED_SCORE_THRESHOLD = 0.82
MIN_FILES_FOR_ASSESSMENT = 3


@dataclass(frozen=True)
class FolderAssessment:
    path: Path
    score: float
    files: int
    subdirs: int
    matching: int
    loose: int

try:
    import git
except ImportError:
    git = None  # type: ignore

TEXT_EXTS = {".txt", ".md", ".markdown", ".rtf"}
CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".sh",
    ".bash",
    ".zsh",
    ".sql",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".webp",
}


def classify_path(path: Path) -> FileType:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return FileType.PDF
    if ext in DOCX_EXTS:
        return FileType.DOCX
    if ext in IMAGE_EXTS:
        return FileType.IMAGE
    if ext in TEXT_EXTS:
        return FileType.TEXT
    if ext in CODE_EXTS:
        return FileType.CODE
    return FileType.OTHER


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in ("*", "?", "["))


def _is_in_app_bundle(path: Path) -> bool:
    return any(part.endswith(".app") for part in path.parts)


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part.startswith(".") for part in rel.parts if part not in (".", ".."))


def _matches_pattern(path: Path, root: Path, pattern: str) -> bool:
    abs_path = path.as_posix()
    try:
        rel_path = path.relative_to(root).as_posix()
    except ValueError:
        rel_path = abs_path

    if _has_glob(pattern):
        return fnmatch.fnmatch(abs_path, pattern) or fnmatch.fnmatch(rel_path, pattern)

    pattern_path = Path(pattern)
    if not pattern_path.is_absolute():
        pattern_path = root / pattern_path
    return path == pattern_path or path.is_relative_to(pattern_path)


def should_ignore(path: Path, root: Path, config: AppConfig) -> bool:
    if not config.scan.include_hidden and _is_hidden(path, root):
        return True
    for pattern in config.scan.ignore_patterns:
        if _matches_pattern(path, root, pattern):
            return True
    return False


def size_limit_for(file_type: FileType, config: AppConfig) -> int:
    limits = config.indexing.file_size_limits
    if file_type == FileType.PDF:
        return limits.pdf
    if file_type == FileType.DOCX:
        return limits.docx
    if file_type == FileType.IMAGE:
        return limits.image
    if file_type in (FileType.TEXT, FileType.CODE):
        return limits.txt
    return limits.default


_FILE_NAMING_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}_[a-z0-9-]+(?:_[a-z0-9-]+)*(?:_v\d+)?\.[a-z0-9]{1,6}"
)


def _matches_expected_naming(name: str) -> bool:
    return bool(_FILE_NAMING_PATTERN.fullmatch(name))


def matches_expected_naming(name: str) -> bool:
    return _matches_expected_naming(name)


def _tokenize_name(name: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", name.lower())
        if len(tok) > 2 and not tok.isdigit()
    }


def _average_name_similarity(files: list[tuple[Path, os.stat_result, FileType, bool]], sample_limit: int = 80) -> float:
    sample = files[:sample_limit]
    tokens = [_tokenize_name(path.name) for path, _, _, _ in sample]
    if len(tokens) < 2:
        return 0.0
    pair_scores = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            a, b = tokens[i], tokens[j]
            if not a or not b:
                continue
            overlap = len(a & b)
            union = len(a | b) or 1
            pair_scores.append(overlap / union)
    if not pair_scores:
        return 0.0
    return float(sum(pair_scores) / len(pair_scores))


def _assess_folder(path: Path, subdirs: list[str], files: list[tuple[Path, os.stat_result, FileType, bool]]) -> FolderAssessment:
    file_count = len(files)
    matching = sum(1 for _, _, _, match in files if match)
    loose = max(0, file_count - matching)
    subdir_count = len(subdirs)
    naming_ratio = matching / file_count if file_count else 0.0
    structure_ratio = subdir_count / (file_count + subdir_count) if (file_count + subdir_count) else 0.0
    name_similarity = _average_name_similarity(files)
    score = round(0.55 * naming_ratio + 0.25 * name_similarity + 0.2 * structure_ratio, 3)
    return FolderAssessment(
        path=path,
        score=score,
        files=file_count,
        subdirs=subdir_count,
        matching=matching,
        loose=loose,
    )


def _average_embedding_similarity_from_paths(
    file_paths: list[str],
    embeddings: dict[str, list[float]],
    sample_limit: int = 60,
) -> float:
    vectors_by_dim: dict[int, list[np.ndarray]] = {}
    for path in file_paths[:sample_limit]:
        vec = embeddings.get(path)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=float)
        vectors_by_dim.setdefault(arr.size, []).append(arr)
    if not vectors_by_dim:
        return 0.0

    def _avg_for_group(group: list[np.ndarray]) -> float:
        if len(group) < 2:
            return 0.0
        group = [g / (np.linalg.norm(g) or 1e-9) for g in group]
        total = 0.0
        count = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total += float(np.dot(group[i], group[j]))
                count += 1
        return total / count if count else 0.0

    scores = [_avg_for_group(group) for group in vectors_by_dim.values()]
    return float(sum(scores) / len(scores))


def _average_name_similarity_from_paths(file_paths: list[str], sample_limit: int = 80) -> float:
    sample = file_paths[:sample_limit]
    tokens = [_tokenize_name(Path(p).name) for p in sample]
    if len(tokens) < 2:
        return 0.0
    pair_scores = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            a, b = tokens[i], tokens[j]
            if not a or not b:
                continue
            overlap = len(a & b)
            union = len(a | b) or 1
            pair_scores.append(overlap / union)
    if not pair_scores:
        return 0.0
    return float(sum(pair_scores) / len(pair_scores))


def _assess_folder_from_paths(
    folder: Path,
    file_paths: list[str],
    embeddings: dict[str, list[float]] | None = None,
) -> FolderAssessment:
    file_count = len(file_paths)
    if file_count == 0:
        return FolderAssessment(path=folder, score=0.0, files=0, subdirs=0, matching=0, loose=0)

    matching = sum(1 for p in file_paths if _matches_expected_naming(Path(p).name))
    loose = max(0, file_count - matching)
    subdirs: set[str] = set()
    for path_str in file_paths:
        rel_parts = Path(path_str).parent.relative_to(folder).parts
        if rel_parts:
            subdirs.add(rel_parts[0])
    subdir_count = len(subdirs)

    naming_ratio = matching / file_count if file_count else 0.0
    structure_ratio = subdir_count / (file_count + subdir_count) if (file_count + subdir_count) else 0.0
    name_similarity = _average_name_similarity_from_paths(file_paths)
    embed_similarity = (
        _average_embedding_similarity_from_paths(file_paths, embeddings or {})
        if embeddings is not None
        else 0.0
    )
    score = round(
        0.55 * naming_ratio + 0.2 * name_similarity + 0.15 * structure_ratio + 0.1 * embed_similarity,
        3,
    )
    return FolderAssessment(
        path=folder,
        score=score,
        files=file_count,
        subdirs=subdir_count,
        matching=matching,
        loose=loose,
    )


def assess_folders_from_paths(
    file_paths: list[str],
    embeddings: dict[str, list[float]] | None = None,
    threshold: float = DEFAULT_ORGANIZED_SCORE_THRESHOLD,
) -> tuple[dict[Path, FolderAssessment], set[Path]]:
    by_folder: dict[Path, list[str]] = {}
    for path_str in file_paths:
        parent = Path(path_str).parent
        by_folder.setdefault(parent, []).append(path_str)

    assessments: dict[Path, FolderAssessment] = {}
    organized: set[Path] = set()
    for folder, paths in by_folder.items():
        if len(paths) < MIN_FILES_FOR_ASSESSMENT:
            continue
        assessment = _assess_folder_from_paths(folder, paths, embeddings=embeddings)
        assessments[folder] = assessment
        if assessment.score >= threshold:
            organized.add(folder)
    return assessments, organized


def _is_ignored_git_repo(path: Path, config: AppConfig) -> bool:
    if git is None:
        return False
    try:
        repo = git.Repo(path)
    except Exception:
        return False
    try:
        remotes = [url for r in repo.remotes for url in r.urls]
    except Exception:
        remotes = []
    hosts = config.scan.git_remote_hosts
    for url in remotes:
        for host in hosts:
            if host in url:
                return True
    return False


def scan_files(
    config: AppConfig,
    organized_threshold: float = DEFAULT_ORGANIZED_SCORE_THRESHOLD,
    on_directory: Callable[[FolderAssessment, str], None] | None = None,
    on_git_repo: Callable[[Path], None] | None = None,
) -> Iterable[FileInfo | None]:
    root = config.scan.root
    follow_symlinks = config.scan.follow_symlinks
    resolved_root = root.resolve()
    for current_root, dirs, files in os.walk(root, followlinks=follow_symlinks):
        current_path = Path(current_root)
        is_git_remote = (current_path / ".git").is_dir() and _is_ignored_git_repo(current_path, config)
        if is_git_remote and on_git_repo:
            try:
                on_git_repo(current_path)
            except Exception:
                pass
        if config.scan.ignore_git_repos and is_git_remote:
            dirs[:] = []
            yield None
            continue
        pruned_dirs = []
        for name in dirs:
            candidate = current_path / name
            if _is_in_app_bundle(candidate):
                continue
            if not follow_symlinks and candidate.is_symlink():
                continue
            if should_ignore(candidate, root, config):
                continue
            pruned_dirs.append(name)
        dirs[:] = pruned_dirs

        file_records: list[tuple[Path, os.stat_result, FileType, bool]] = []
        for name in files:
            path = current_path / name
            if _is_in_app_bundle(path):
                continue
            if should_ignore(path, root, config):
                continue
            if not follow_symlinks and path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            file_type = classify_path(path)
            if file_type == FileType.OTHER:
                continue
            if stat.st_size > size_limit_for(file_type, config):
                continue
            matches_pattern = _matches_expected_naming(path.name)
            file_records.append((path, stat, file_type, matches_pattern))

        assessment = _assess_folder(current_path, pruned_dirs, file_records)
        is_root_dir = current_path.resolve() == resolved_root
        organized_like = (
            not is_root_dir
            and assessment.files >= MIN_FILES_FOR_ASSESSMENT
            and assessment.score >= organized_threshold
        )

        if on_directory and assessment.files >= MIN_FILES_FOR_ASSESSMENT:
            status = "organized" if organized_like else "needs_attention"
            on_directory(assessment, status)

        for path, stat, file_type, _ in file_records:
            yield FileInfo(
                path=path, size=stat.st_size, mtime=stat.st_mtime, file_type=file_type
            )
