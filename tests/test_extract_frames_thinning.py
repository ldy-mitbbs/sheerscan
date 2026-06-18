"""Frame extraction honors INSPECTOR_INTERVAL even when ffmpeg's `fps` filter
fails to thin a dense-GOP source.

Broadcast transport streams often carry keyframes *denser* than the configured
interval (e.g. a 2s GOP vs an 8s interval). `fps=1/interval` silently passes all
of them through on a full file, so the coarse VLM pass would pay for ~4x the
frames the interval intends. `_extract_frames_ffmpeg` subsamples the keyframe
list back onto the interval grid; these tests pin that behavior (ffmpeg is
mocked so they run anywhere, no real decode).
"""
from pathlib import Path

from sheerscan.inspector import VideoInspector


def _make_files(temp_dir, n):
    files = []
    for i in range(1, n + 1):
        p = Path(temp_dir) / f"frame_{i:05d}.jpg"
        p.write_bytes(b"x")
        files.append(p)
    return sorted(files)


def test_dense_keyframe_source_thinned_to_interval_grid(tmp_path, monkeypatch):
    insp = VideoInspector()
    monkeypatch.setattr(insp, "_video_duration_seconds", lambda vp: 8000.0)
    # fps failed to thin -> ~4000 keyframes for an 8000s / 8s = 1000-frame grid
    monkeypatch.setattr(insp, "_run_ffmpeg_fps",
                        lambda vp, td, iv, keyframe_only: _make_files(td, 4000))

    frames = insp._extract_frames_ffmpeg("v.ts", str(tmp_path), 8)

    # thinned onto the 8s grid (~1000), not 4000
    assert 950 <= len(frames) <= 1050
    # grid timing preserved
    assert frames[0]["seconds"] == 0.0
    assert frames[1]["seconds"] == 8.0
    # dropped JPEGs were removed; only the kept ones remain on disk
    assert len(list(tmp_path.glob("frame_*.jpg"))) == len(frames)


def test_well_behaved_source_not_thinned(tmp_path, monkeypatch):
    insp = VideoInspector()
    monkeypatch.setattr(insp, "_video_duration_seconds", lambda vp: 8000.0)
    # fps thinned correctly -> ~expected count; must be left untouched
    monkeypatch.setattr(insp, "_run_ffmpeg_fps",
                        lambda vp, td, iv, keyframe_only: _make_files(td, 1000))

    frames = insp._extract_frames_ffmpeg("v.ts", str(tmp_path), 8)
    assert len(frames) == 1000


def test_sparse_keyframes_fall_back_to_full_decode(tmp_path, monkeypatch):
    insp = VideoInspector()
    monkeypatch.setattr(insp, "_video_duration_seconds", lambda vp: 8000.0)
    calls = []

    def fake_fps(vp, td, iv, keyframe_only):
        calls.append(keyframe_only)
        # keyframes too sparse (<0.9*expected) -> full decode yields the grid
        return _make_files(td, 100 if keyframe_only else 1000)

    monkeypatch.setattr(insp, "_run_ffmpeg_fps", fake_fps)

    frames = insp._extract_frames_ffmpeg("v.ts", str(tmp_path), 8)
    assert calls == [True, False]      # tried keyframe-only, then full decode
    assert len(frames) == 1000


def test_scene_grid_keeps_grid_and_cuts_drops_rest(tmp_path, monkeypatch):
    """_extract_scene_grid keeps grid-aligned frames + keyframes near scene cuts,
    and unlinks the rest."""
    from sheerscan.inspector import VideoInspector
    insp = VideoInspector()

    # synthetic keyframes every 2s out to 40s, with grid-aligned ones protected
    INTERVAL = 8
    frames = []
    for i in range(21):
        sec = i * 2.0
        p = tmp_path / f"frame_{i+1:05d}.jpg"
        p.write_bytes(b"x")
        f = {"file_path": p, "seconds": sec, "timestamp_str": str(sec),
             "id": f"frame_{i+1:04d}"}
        if sec % INTERVAL == 0:          # 0,8,16,24,32,40 -> grid
            f["protected_grid"] = True
        frames.append(f)
    monkeypatch.setattr(insp, "_extract_keyframes_dense",
                        lambda vp, td, iv: frames)

    cuts = [10.0, 26.0]                  # off-grid shot boundaries
    kept = insp._extract_scene_grid("v.ts", str(tmp_path), INTERVAL, cuts)
    kept_secs = {f["seconds"] for f in kept}

    # every grid frame survives
    for g in (0.0, 8.0, 16.0, 24.0, 32.0, 40.0):
        assert g in kept_secs, g
    # the keyframe at each cut survives
    assert 10.0 in kept_secs and 26.0 in kept_secs
    # a frame far from grid and cuts is dropped AND its JPEG unlinked
    assert 4.0 not in kept_secs
    assert 20.0 not in kept_secs
    assert not (tmp_path / "frame_00003.jpg").exists()   # sec=4.0
    assert not (tmp_path / "frame_00011.jpg").exists()    # sec=20.0
    # kept frames are re-indexed contiguously
    assert [f["id"] for f in kept] == [f"frame_{i+1:04d}" for i in range(len(kept))]
