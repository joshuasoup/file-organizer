from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable

from fileorg.config import AppConfig
from fileorg.indexer.types import FileInfo, FileType
from fileorg.indexer.types import FileInfo, FileType

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


def scan_files(config: AppConfig) -> Iterable[FileInfo | None]:
    root = config.scan.root
    follow_symlinks = config.scan.follow_symlinks
    for current_root, dirs, files in os.walk(root, followlinks=follow_symlinks):
        current_path = Path(current_root)
        if (
            config.scan.ignore_git_repos
            and (current_path / ".git").is_dir()
            and _is_ignored_git_repo(current_path, config)
        ):
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
            yield FileInfo(
                path=path, size=stat.st_size, mtime=stat.st_mtime, file_type=file_type
            )
