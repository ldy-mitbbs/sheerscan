"""Durable, portable ground-truth corpus for the stockings video inspector.

The corpus decouples human labels from any single inspection job so prompt /
model / post-processing changes can be regression-tested against a stable set
of human-annotated moments.

Two layers, deliberately split for privacy:

* **Committed** (versioned with the repo, under ``tests/inspector_corpus/``):
  ``manifest.jsonl`` — one labeled example per row, and ``baseline.json`` —
  the recorded headline metrics the regression test asserts against. Neither
  contains real filenames: each example carries a hashed ``video_id`` only.

* **Local** (under ``~/sheerscan-local/corpus/``, gitignored): the actual frame
  JPEGs (named by content hash) plus ``videos.json`` mapping ``video_id`` back
  to the real ``container_path`` so scoring can line corpus examples up with a
  run's detections on this machine.

A corpus example::

    {
      "id": "<stable hash>",          # (video_id, kind, seconds) — re-harvest updates in place
      "video_id": "<sha1(container_path)[:16]>",
      "seconds": 1956.0,
      "timestamp": "00:32:36",
      "label": "positive" | "negative",
      "kind": "event" | "frame",      # event = mpv mark, frame = reviewed AI detection
      "frame_sha256": "<hex>" | null,
      "source": "mark" | "feedback:false_positive" | "manual_label" | ...,
      "reason": "",                   # model/user rationale, when known
      "note": "",
      "created_at": 1716900000,
      "updated_at": 1716900000,
    }
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .runtime import get_setting, get_local_video_dir

POSITIVE = "positive"
NEGATIVE = "negative"

# Map detection-feedback labels to corpus labels. ``unsure`` is intentionally
# absent — ambiguous feedback is skipped rather than guessed.
FEEDBACK_LABEL_MAP = {
    "love": POSITIVE,
    "like": POSITIVE,
    "ok": POSITIVE,
    "false_positive": NEGATIVE,
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------- locations

def corpus_manifest_dir() -> Path:
    """Committed manifest/baseline dir. Overridable via INSPECTOR_CORPUS_DIR."""
    override = get_setting("INSPECTOR_CORPUS_DIR", None)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "tests" / "inspector_corpus"


def manifest_path() -> Path:
    return corpus_manifest_dir() / "manifest.jsonl"


def baseline_path() -> Path:
    return corpus_manifest_dir() / "baseline.json"


def local_corpus_dir() -> Path:
    """Local (gitignored) dir holding frame JPEGs + the video_id map."""
    return get_local_video_dir() / "corpus"


def frames_dir() -> Path:
    return local_corpus_dir() / "frames"


def videos_map_path() -> Path:
    return local_corpus_dir() / "videos.json"


def inspections_dir() -> Path:
    return get_local_video_dir() / "inspections"


# ---------------------------------------------------------------- hashing

def video_id_for(container_path: str) -> str:
    return hashlib.sha1((container_path or "").encode("utf-8")).hexdigest()[:16]


def _example_id(video_id: str, kind: str, seconds: float) -> str:
    key = f"{video_id}:{kind}:{round(float(seconds or 0.0), 2)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- manifest I/O

def load_manifest() -> list[dict]:
    path = manifest_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def save_manifest(rows: list[dict]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.get("video_id") or "", float(r.get("seconds") or 0.0), r.get("kind") or ""))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_videos_map() -> dict:
    path = videos_map_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_videos_map(mapping: dict) -> None:
    path = videos_map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def container_path_for(video_id: str) -> Optional[str]:
    entry = load_videos_map().get(video_id)
    if isinstance(entry, dict):
        return entry.get("container_path")
    return None


# ---------------------------------------------------------------- frame store

def _store_frame(src: Path) -> Optional[str]:
    """Copy a frame into the content-addressed local store; return its sha256."""
    digest = hash_file(src)
    if not digest:
        return None
    dest = frames_dir() / f"{digest}{src.suffix.lower() if src.suffix.lower() in _IMAGE_EXTS else '.jpg'}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(src.read_bytes())
        except OSError:
            return None
    return digest


def frame_path_for(frame_sha256: str) -> Optional[Path]:
    if not frame_sha256:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = frames_dir() / f"{frame_sha256}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------- harvest

def harvest(*, verbose: bool = False) -> dict:
    """Consolidate manual marks + detection feedback into the corpus.

    Idempotent: re-running updates existing rows (e.g. a relabeled frame) and
    refreshes the frame store / video map rather than duplicating.
    """
    from .inspector import list_manual_marks  # late import: heavy module

    rows = {row["id"]: row for row in load_manifest() if row.get("id")}
    videos = load_videos_map()
    now = int(time.time())
    stats = {"marks": 0, "feedback": 0, "frames_copied": 0, "skipped": 0, "videos": 0}

    def _touch_video(container_path: str, video_file: str | None) -> str:
        vid = video_id_for(container_path)
        if vid not in videos:
            videos[vid] = {"container_path": container_path, "video_file": video_file}
            stats["videos"] += 1
        elif video_file and not videos[vid].get("video_file"):
            videos[vid]["video_file"] = video_file
        return vid

    def _upsert(row: dict) -> None:
        existing = rows.get(row["id"])
        if existing:
            row["created_at"] = existing.get("created_at", now)
            # keep a stored frame hash if the fresh source lacks one
            if not row.get("frame_sha256") and existing.get("frame_sha256"):
                row["frame_sha256"] = existing["frame_sha256"]
        rows[row["id"]] = row

    # 1) mpv manual marks -> positive/negative "event" examples
    marks_root = inspections_dir() / "manual_marks"
    for mark in list_manual_marks():
        cp = mark.get("container_path")
        if not cp:
            stats["skipped"] += 1
            continue
        label = (mark.get("label") or POSITIVE).strip().lower()
        if label not in (POSITIVE, NEGATIVE):
            label = POSITIVE
        seconds = float(mark.get("seconds") or 0.0)
        vid = _touch_video(cp, mark.get("filename"))
        frame_sha = None
        shot = (mark.get("screenshot_file") or "").strip()
        if shot:
            src = marks_root / shot
            if src.exists():
                frame_sha = _store_frame(src)
                if frame_sha:
                    stats["frames_copied"] += 1
        _upsert({
            "id": _example_id(vid, "event", seconds),
            "video_id": vid,
            "seconds": seconds,
            "timestamp": mark.get("timestamp") or "",
            "label": label,
            "kind": "event",
            "frame_sha256": frame_sha,
            "source": mark.get("source") or "mark",
            "reason": mark.get("note") or "",
            "note": mark.get("note") or "",
            "updated_at": now,
            "created_at": now,
        })
        stats["marks"] += 1

    # 2) reviewed AI detections (feedback.jsonl) -> positive/negative "frame" examples
    feedback_path = inspections_dir() / "feedback.jsonl"
    if feedback_path.exists():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fb_label = (rec.get("feedback") or {}).get("label")
            label = FEEDBACK_LABEL_MAP.get(fb_label)
            if not label:
                stats["skipped"] += 1
                continue
            det = rec.get("detection") or {}
            cp = rec.get("container_path")
            seconds = det.get("seconds")
            if not cp or seconds is None:
                stats["skipped"] += 1
                continue
            seconds = float(seconds)
            vid = _touch_video(cp, rec.get("video_file"))
            frame_sha = None
            image_file = (det.get("image_file") or "").strip()
            job_id = rec.get("job_id")
            if image_file and job_id:
                src = inspections_dir() / job_id / image_file
                if src.exists():
                    frame_sha = _store_frame(src)
                    if frame_sha:
                        stats["frames_copied"] += 1
            stage = det.get("source") or (
                "coarse_fallback" if det.get("fine_empty_fallback")
                else ("fine" if str(det.get("frame_id") or "").startswith("frame_fine") else "coarse")
            )
            _upsert({
                "id": _example_id(vid, "frame", seconds),
                "video_id": vid,
                "seconds": seconds,
                "timestamp": det.get("timestamp") or "",
                "label": label,
                "kind": "frame",
                "frame_sha256": frame_sha,
                "source": f"feedback:{fb_label}",
                "confidence": str(det.get("confidence") or "").lower() or None,
                "stage": stage,
                "reason": det.get("reason") or "",
                "note": (rec.get("feedback") or {}).get("note") or "",
                "updated_at": now,
                "created_at": now,
            })
            stats["feedback"] += 1

    save_manifest(list(rows.values()))
    _save_videos_map(videos)
    stats["total_examples"] = len(rows)
    if verbose:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


# ---------------------------------------------------------------- queries

def set_label(example_id: str, label: str) -> Optional[dict]:
    label = (label or "").strip().lower()
    if label not in (POSITIVE, NEGATIVE):
        raise ValueError(f"invalid label: {label!r}")
    rows = load_manifest()
    updated = None
    for row in rows:
        if row.get("id") == example_id:
            row["label"] = label
            row["updated_at"] = int(time.time())
            if row.get("source", "").startswith("feedback"):
                row["source"] = "manual_label"
            updated = row
            break
    if updated:
        save_manifest(rows)
    return updated


def corpus_by_video() -> dict[str, dict]:
    """Group the manifest by video_id into {positives, negatives} buckets."""
    out: dict[str, dict] = {}
    for row in load_manifest():
        vid = row.get("video_id")
        if not vid:
            continue
        bucket = out.setdefault(vid, {"positives": [], "negatives": [], "container_path": container_path_for(vid)})
        if row.get("label") == NEGATIVE:
            bucket["negatives"].append(row)
        else:
            bucket["positives"].append(row)
    return out


def corpus_stats() -> dict:
    rows = load_manifest()
    videos = {r.get("video_id") for r in rows}
    positives = [r for r in rows if r.get("label") != NEGATIVE]
    negatives = [r for r in rows if r.get("label") == NEGATIVE]
    return {
        "total_examples": len(rows),
        "videos": len(videos),
        "positives": len(positives),
        "negatives": len(negatives),
        "with_frames": sum(1 for r in rows if r.get("frame_sha256")),
    }
