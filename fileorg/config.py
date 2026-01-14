from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fileorg import paths

MB = 1024 * 1024
HOME_PATH = Path.home()


def _expand_user(value: str | Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    return Path(expanded)


def _expand_pattern(pattern: str) -> str:
    return os.path.expandvars(os.path.expanduser(pattern))


def _contract_home(value: str | Path) -> str:
    value_str = str(value)
    home_str = str(HOME_PATH)
    if value_str == home_str:
        return "~"
    if value_str.startswith(home_str + "/"):
        return "~" + value_str[len(home_str) :]
    return value_str


def default_ignore_patterns() -> list[str]:
    return [
        "~/Applications",
        "~/Music",
        "~/Pictures",
        "~/Library",
        "~/Movies",
        "~/.Trash",
        "/Volumes/*",
        "/Network/*",
        "**/*.app",
        "**/node_modules",
        "**/build",
        "**/dist",
        "**/__pycache__",
        "**/.venv",
        "**/venv",
        "**/.mypy_cache",
        "**/.pytest_cache",
        "**/.ruff_cache",
        "**/.tox",
        "**/.cache",
        "**/.git",
        "**/.hg",
        "**/.svn",
    ]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class FileSizeLimits(ConfigModel):
    pdf: int = 100 * MB
    docx: int = 50 * MB
    txt: int = 200 * MB
    image: int = 10 * MB
    default: int = 5 * MB


class IndexingConfig(ConfigModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_chunks_per_file: int = 200
    file_size_limits: FileSizeLimits = Field(default_factory=FileSizeLimits)


class EmbeddingsConfig(ConfigModel):
    text_model: str = "BAAI/bge-m3"
    image_model: str = "ViT-B/32"
    text_provider: str = "huggingface"
    image_provider: str = "clip"


class ScanConfig(ConfigModel):
    root: Path = Field(default_factory=lambda: HOME_PATH)
    follow_symlinks: bool = False
    include_hidden: bool = False
    ignore_patterns: list[str] = Field(default_factory=default_ignore_patterns)
    ignore_git_repos: bool = True
    git_remote_hosts: list[str] = Field(default_factory=lambda: ["github.com"])

    @field_validator("root", mode="before")
    @classmethod
    def _normalize_root(cls, value: str | Path) -> Path:
        return _expand_user(value)

    @field_validator("ignore_patterns", mode="before")
    @classmethod
    def _normalize_patterns(cls, value: Iterable[str] | None) -> list[str]:
        if value is None:
            return default_ignore_patterns()
        return [_expand_pattern(pattern) for pattern in value]


class StorageConfig(ConfigModel):
    base_dir: Path = Field(default_factory=paths.app_dir)

    @field_validator("base_dir", mode="before")
    @classmethod
    def _normalize_base_dir(cls, value: str | Path) -> Path:
        return _expand_user(value)


class AppConfig(ConfigModel):
    scan: ScanConfig = Field(default_factory=ScanConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def chroma_dir(self) -> Path:
        return self.storage.base_dir / "chroma"

    def metadata_db_path(self) -> Path:
        return self.storage.base_dir / "metadata.sqlite3"


def render_config_toml(config: AppConfig) -> str:
    ignore_patterns = [
        f'  "{_contract_home(pattern)}"' for pattern in config.scan.ignore_patterns
    ]
    ignore_block = ",\n".join(ignore_patterns)
    lines = [
        "[scan]",
        f'root = "{_contract_home(config.scan.root)}"',
        f"follow_symlinks = {str(config.scan.follow_symlinks).lower()}",
        f"include_hidden = {str(config.scan.include_hidden).lower()}",
        "ignore_patterns = [",
        ignore_block,
        "]",
        f"ignore_git_repos = {str(config.scan.ignore_git_repos).lower()}",
        "git_remote_hosts = [",
        *[
            f'  "{host}"' + ("," if i < len(config.scan.git_remote_hosts) - 1 else "")
            for i, host in enumerate(config.scan.git_remote_hosts)
        ],
        "]",
        "",
        "[indexing]",
        f"chunk_size = {config.indexing.chunk_size}",
        f"chunk_overlap = {config.indexing.chunk_overlap}",
        f"max_chunks_per_file = {config.indexing.max_chunks_per_file}",
        "",
        "[indexing.file_size_limits]",
        f"pdf = {config.indexing.file_size_limits.pdf}",
        f"docx = {config.indexing.file_size_limits.docx}",
        f"txt = {config.indexing.file_size_limits.txt}",
        f"image = {config.indexing.file_size_limits.image}",
        f"default = {config.indexing.file_size_limits.default}",
        "",
        "[embeddings]",
        f'text_model = "{config.embeddings.text_model}"',
        f'image_model = "{config.embeddings.image_model}"',
        f'text_provider = "{config.embeddings.text_provider}"',
        f'image_provider = "{config.embeddings.image_provider}"',
        "",
        "[storage]",
        f'base_dir = "{_contract_home(config.storage.base_dir)}"',
        "",
    ]
    return "\n".join(lines)


def load_config(config_file: Path | None = None, create: bool = True) -> AppConfig:
    path = config_file or paths.config_path()
    if path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return AppConfig.model_validate(data)

    config = AppConfig()
    if create:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_config_toml(config), encoding="utf-8")
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write config at {path}. Set FILEORG_HOME or FILEORG_CONFIG to a writable directory."
            ) from exc
    return config
