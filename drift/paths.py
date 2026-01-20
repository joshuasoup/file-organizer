from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "Drift"


def app_dir() -> Path:
    env_home = os.environ.get("DRIFT_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / "Library" / "Application Support" / APP_NAME


def config_path() -> Path:
    env_config = os.environ.get("DRIFT_CONFIG")
    if env_config:
        return Path(env_config).expanduser()
    return app_dir() / "config.toml"


def ensure_app_dir() -> Path:
    path = app_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
