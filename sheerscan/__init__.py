"""sheerscan — VLM hosiery-detection pipeline, extractable from its host app.

Public surface (filled in across extraction phases):

    from sheerscan import configure
    from sheerscan.inspector import VideoInspector, VideoInspectorJobManager

Standalone usage reads config from env / ``~/.config/sheerscan/config.json``.
A host app injects its own config + path mapping via :func:`configure`.
"""
from __future__ import annotations

from .runtime import (
    configure,
    SettingsProvider,
    PathMapper,
    EnvSettings,
    IdentityPathMapper,
)

__all__ = [
    "configure",
    "SettingsProvider",
    "PathMapper",
    "EnvSettings",
    "IdentityPathMapper",
]

__version__ = "0.1.0"
