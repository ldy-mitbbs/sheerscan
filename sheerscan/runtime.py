"""Injectable runtime: config + path mapping.

This is the seam that lets the package run **standalone** (the defaults below
read only environment variables, with no container-path mapping) while a
host application can inject its own settings store and path mapper
via :func:`configure` so the two share one config file and behave identically.

Every pipeline module imports the module-level helpers (``get_setting``,
``get_secret``, ``to_host_path`` …) from here instead of reaching into the host
app. The helpers delegate to whatever providers are currently configured, so
call sites never change when the host swaps the backend in.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SettingsProvider(Protocol):
    """Reads tuning knobs (non-secret) and secrets (API keys)."""

    def get_setting(self, name: str, default=None): ...
    def get_secret(self, name: str) -> Optional[str]: ...
    def get_local_dir(self) -> Path: ...


@runtime_checkable
class PathMapper(Protocol):
    """Translates between the path a downloader/container sees and the local path."""

    def to_host(self, path: str) -> str: ...
    def to_container(self, path: str) -> str: ...


# ---- standalone defaults -------------------------------------------------
class EnvSettings:
    """Default provider: env var wins, then ``~/.config/sheerscan/config.json``,
    then the recommended pipeline defaults below.

    Secrets and plain settings share the same lookup here; a host app that
    separates them injects its own provider instead.
    """

    # The settled, recommended pipeline: coarse-only VLM pass + the semantic
    # reason filter as the sole judge (see README "为什么长这样"). The inline
    # ``get_setting(name, "0")`` call sites in inspector.py still default the
    # *legacy* hybrid behaviour off-by-default for embedders; here, for the
    # standalone install, we flip the two knobs that make a fresh `pip install`
    # run the pipeline the project actually recommends. Env var or config.json
    # still override. A host app that injects its own provider never sees these.
    _RECOMMENDED = {
        "INSPECTOR_SKIP_FINE_PASS": "1",       # coarse-only (the fine pass hurt recall+precision)
        "INSPECTOR_REASON_FILTER": "1",        # semantic filter is the sole judge
        "INSPECTOR_SIMPLE_COARSE_PROMPT": "1",  # the flood-resistant simple prompt
    }

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or (
            Path.home() / ".config" / "sheerscan" / "config.json"
        )

    def _file(self) -> dict:
        try:
            return json.loads(self._config_path.read_text("utf-8"))
        except Exception:
            return {}

    def get_setting(self, name: str, default=None):
        if name in os.environ:
            return os.environ[name]
        file_val = self._file().get(name)
        if file_val is not None:
            return file_val
        if name in self._RECOMMENDED:
            return self._RECOMMENDED[name]
        return default

    def get_secret(self, name: str) -> Optional[str]:
        return os.environ.get(name) or self._file().get(name) or None

    def get_local_dir(self) -> Path:
        raw = self.get_setting("SHEERSCAN_LOCAL_DIR", None)
        if raw:
            return Path(raw).expanduser()
        return Path.home() / "sheerscan-local"


class IdentityPathMapper:
    """Default mapper: paths are already local; no translation."""

    def to_host(self, path: str) -> str:
        return path

    def to_container(self, path: str) -> str:
        return path


# ---- active providers ----------------------------------------------------
_settings: SettingsProvider = EnvSettings()
_pathmap: PathMapper = IdentityPathMapper()


def configure(
    *,
    settings: Optional[SettingsProvider] = None,
    pathmap: Optional[PathMapper] = None,
) -> None:
    """Install host-supplied providers. Call once at startup; omit to keep defaults."""
    global _settings, _pathmap
    if settings is not None:
        _settings = settings
    if pathmap is not None:
        _pathmap = pathmap


# ---- module-level helpers the pipeline imports ---------------------------
def get_setting(name: str, default=None):
    return _settings.get_setting(name, default)


def get_secret(name: str) -> Optional[str]:
    return _settings.get_secret(name)


def get_local_video_dir() -> Path:
    return _settings.get_local_dir()


def to_host_path(path: str) -> str:
    return _pathmap.to_host(path)


def to_container_path(path: str) -> str:
    return _pathmap.to_container(path)
