"""runtime.scene_cuts must hand the host hook a str even when given a Path
(the inspector passes a Path video_path; a host PathMapper that does string ops
on it would otherwise raise and silently yield None)."""
from pathlib import Path
import sheerscan.runtime as rt


def test_scene_cuts_coerces_path_to_str(monkeypatch):
    seen = {}

    class PM:
        def to_host(self, p): return p
        def to_container(self, p): return p
        def scene_cuts(self, p):
            seen["type"] = type(p).__name__
            return [1.0, 2.0]

    monkeypatch.setattr(rt, "_pathmap", PM())
    out = rt.scene_cuts(Path("/data/x.mkv"))
    assert seen["type"] == "str"      # hook received a str, not a PosixPath
    assert out == [1.0, 2.0]


def test_scene_cuts_none_without_hook(monkeypatch):
    class PM:
        def to_host(self, p): return p
        def to_container(self, p): return p
    monkeypatch.setattr(rt, "_pathmap", PM())
    assert rt.scene_cuts("/data/x.mkv") is None
