"""Offline replay harness for tuning the stockings video inspector.

Two tiers, by cost:

* **Tier A — post-processing replay (free, deterministic).** Reconstructs a job's
  final detections from its saved ``trace.json`` by re-running the *current*
  filtering / evidence / dedupe / windowing code (the pure helpers in
  ``video_inspector``) over the recorded raw model verdicts. Lets you tune all
  the post-processing knobs and see recall/precision deltas across the whole
  corpus with zero API calls. Window/sampling changes are NOT reflected — the
  trace only records the segments that were actually sent — so those need Tier B.

* **Tier B — frame-classification replay (costs API, fixed input).** Re-sends the
  corpus's labeled frames to a candidate model/prompt via ``inspect_batch`` and
  measures how cleanly it separates positive from negative frames. Fixed inputs
  → clean A/B between prompts/models without re-extracting video.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import corpus
from .runtime import get_setting, get_local_video_dir
from .inspector import (
    VideoInspector,
    is_truthy_setting,
    get_float_setting,
    keep_coarse_detection,
    keep_fine_detection,
    fine_video_detection_seconds,
    finalize_detections,
    annotate_scene_support,
    filter_by_scene_support,
    get_int_setting,
    evaluate_against_corpus,
)

COARSE_STAGES = {"coarse_screenshot", "screenshot_coarse"}
SCREENSHOT_FINE_STAGES = {"screenshot_fine"}
VIDEO_FINE_STAGES = {"fine_video"}


def replay_config(overrides: Optional[dict] = None) -> dict:
    """Snapshot the post-processing knobs (current settings + optional overrides)."""
    cfg = {
        "extreme_recall": is_truthy_setting(get_setting("INSPECTOR_EXTREME_RECALL", "0")),
        "require_concrete_evidence": is_truthy_setting(get_setting("INSPECTOR_STRICT_EVIDENCE_FILTER", "0")),
        "keep_low_fine": is_truthy_setting(get_setting("INSPECTOR_KEEP_LOW_FINE", "1")),
        "keep_coarse_on_fine_empty": is_truthy_setting(get_setting("INSPECTOR_KEEP_COARSE_ON_FINE_EMPTY", "1")),
        "exclude_male_subject": is_truthy_setting(get_setting("INSPECTOR_EXCLUDE_MALE_SUBJECT", "1")),
        "rep_offset": get_float_setting("INSPECTOR_FINE_VIDEO_REPRESENTATIVE_OFFSET_SECONDS", 4.0, min_value=0.0, max_value=20.0),
        "dedup_window": float(get_setting("INSPECTOR_DEDUP_WINDOW", 3.0) or 3.0),
        "min_scene_support": get_int_setting("INSPECTOR_MIN_SCENE_SUPPORT", 1, min_value=1, max_value=20),
        "scene_support_window": get_float_setting("INSPECTOR_SCENE_SUPPORT_WINDOW_SECONDS", 10.0, min_value=1.0, max_value=120.0),
        "skip_fine": is_truthy_setting(get_setting("INSPECTOR_SKIP_FINE_PASS", "0")),
        "drop_weak_coarse": is_truthy_setting(get_setting("INSPECTOR_DROP_WEAK_COARSE", "0")),
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def _coarse_fallback(coarse: list[dict]) -> list[dict]:
    out = []
    for det in coarse:
        item = dict(det)
        item["source"] = item.get("source") or "coarse_fallback"
        item["needs_review"] = True
        item["fine_empty_fallback"] = True
        out.append(item)
    return out


def replay_trace_postprocess(trace: list[dict], cfg: dict) -> list[dict]:
    """Reconstruct a run's final detections from a trace under post-processing ``cfg``.

    Mirrors run_inspection: coarse stages seed candidates, fine stages (screenshot
    or video) produce detections; if a fine pass ran but kept nothing the coarse
    candidates are preserved (when enabled); finally sort + temporal dedupe.
    """
    coarse: list[dict] = []
    fine: list[dict] = []
    saw_fine_stage = False

    for entry in trace or []:
        stage = entry.get("stage")
        if entry.get("skipped"):
            if stage in (SCREENSHOT_FINE_STAGES | VIDEO_FINE_STAGES):
                saw_fine_stage = True
            continue
        parsed = entry.get("parsed_detections") or []
        frames_by_id = {f.get("frame_id"): f for f in (entry.get("frames") or [])}

        if stage in COARSE_STAGES:
            hybrid = stage == "coarse_screenshot"
            for det in parsed:
                if not isinstance(det, dict):
                    continue
                if not keep_coarse_detection(det, extreme_recall=cfg["extreme_recall"], hybrid=hybrid, is_two_pass=True, exclude_male_subject=cfg["exclude_male_subject"], drop_weak_coarse=cfg.get("drop_weak_coarse", False)):
                    continue
                fr = frames_by_id.get(det.get("frame_id"))
                sec = (fr or {}).get("seconds")
                if sec is None:
                    continue
                d = dict(det)
                d["seconds"] = float(sec)
                d["source_stage"] = stage
                coarse.append(d)

        elif stage in SCREENSHOT_FINE_STAGES:
            saw_fine_stage = True
            for det in parsed:
                if not isinstance(det, dict):
                    continue
                if not keep_fine_detection(det, extreme_recall=cfg["extreme_recall"],
                                           require_concrete_evidence=cfg["require_concrete_evidence"],
                                           keep_low_fine=cfg["keep_low_fine"],
                                           exclude_male_subject=cfg["exclude_male_subject"]):
                    continue
                fr = frames_by_id.get(det.get("frame_id"))
                sec = (fr or {}).get("seconds")
                if sec is None:
                    continue
                d = dict(det)
                d["seconds"] = float(sec)
                d["source_stage"] = stage
                fine.append(d)

        elif stage in VIDEO_FINE_STAGES:
            saw_fine_stage = True
            start = float(entry.get("start_seconds") or 0.0)
            end = float(entry.get("end_seconds") or start)
            duration = float(entry.get("duration") or max(0.0, end - start))
            for det in parsed:
                if not isinstance(det, dict):
                    continue
                if not keep_fine_detection(det, extreme_recall=cfg["extreme_recall"],
                                           require_concrete_evidence=cfg["require_concrete_evidence"],
                                           keep_low_fine=cfg["keep_low_fine"],
                                           exclude_male_subject=cfg["exclude_male_subject"]):
                    continue
                abs_sec, src_sec = fine_video_detection_seconds(det.get("time") or "00:00", start, duration, cfg["rep_offset"])
                d = dict(det)
                d["seconds"] = abs_sec
                d["model_seconds"] = src_sec
                d["source_stage"] = stage
                fine.append(d)

    if cfg.get("skip_fine"):
        # Coarse-only policy: ignore the fine stage entirely and keep the
        # (already false-positive/male-filtered) coarse candidates.
        visual = coarse
    elif saw_fine_stage:
        if fine:
            visual = fine
        elif coarse and cfg["keep_coarse_on_fine_empty"]:
            visual = _coarse_fallback(coarse)
        else:
            visual = []
    else:
        # single-pass (no fine stage in the trace) -> coarse candidates are the result
        visual = coarse

    # Temporal corroboration (route 3, opt-in): score each detection by nearby
    # coarse candidates and drop weakly-supported ones (high-confidence protected).
    min_support = cfg.get("min_scene_support", 1)
    if min_support and min_support > 1 and visual:
        annotate_scene_support(visual, coarse, cfg.get("scene_support_window", 10.0))
        visual = filter_by_scene_support(visual, min_support=min_support)

    return finalize_detections(visual, dedup_window=cfg["dedup_window"])


# ---------------------------------------------------------------- job discovery

def _iter_job_traces():
    """Yield (job_id, container_path, trace) for every job with a trace.json."""
    root = get_local_video_dir() / "inspections"
    if not root.exists():
        return
    for job_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trace_path = job_dir / "trace.json"
        if not trace_path.exists():
            continue
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(trace, list):
            continue
        container_path = None
        result_path = job_dir / "result.json"
        if result_path.exists():
            try:
                container_path = json.loads(result_path.read_text(encoding="utf-8")).get("container_path")
            except (OSError, json.JSONDecodeError):
                pass
        if not container_path:
            status_path = job_dir / "job_status.json"
            if status_path.exists():
                try:
                    container_path = json.loads(status_path.read_text(encoding="utf-8")).get("container_path")
                except (OSError, json.JSONDecodeError):
                    pass
        yield job_dir.name, container_path, trace


def replay_postprocess_over_corpus(*, overrides: Optional[dict] = None,
                                   tolerance_seconds: float = 30.0,
                                   merge_window_seconds: float = 8.0) -> dict:
    """Tier A: replay every traced job and score the result against the corpus.

    When several jobs cover the same video the one yielding the most detections
    wins (the most complete run), so a partial/aborted re-run can't shadow a good one.
    """
    cfg = replay_config(overrides)
    by_video = corpus.corpus_by_video()
    cp_to_vid = {bucket.get("container_path"): vid for vid, bucket in by_video.items() if bucket.get("container_path")}

    detections_by_video: dict[str, list[dict]] = {}
    jobs_used: dict[str, str] = {}
    for job_id, container_path, trace in _iter_job_traces():
        vid = cp_to_vid.get(container_path)
        if not vid:
            continue
        dets = replay_trace_postprocess(trace, cfg)
        if len(dets) >= len(detections_by_video.get(vid, [])):
            detections_by_video[vid] = dets
            jobs_used[vid] = job_id

    metrics = evaluate_against_corpus(
        detections_by_video, by_video,
        tolerance_seconds=tolerance_seconds, merge_window_seconds=merge_window_seconds,
    )
    metrics["mode"] = "postprocess"
    metrics["config"] = cfg
    metrics["jobs_used"] = jobs_used
    return metrics


# ---------------------------------------------------------------- Tier B

def replay_frames_classification(*, model: Optional[str] = None,
                                 is_coarse: bool = True,
                                 prompt_override: Optional[str] = None,
                                 overrides: Optional[dict] = None,
                                 batch_size: int = 6) -> dict:
    """Tier B: classify the corpus's labeled frames with a candidate model/prompt.

    Sends each labeled frame (positive/negative) through ``inspect_batch`` and
    applies the same keep filter the live pipeline would, then reports how well
    the model+prompt+filter separates positives from negatives. Costs API calls,
    but only over the (small) labeled frame set with fixed inputs.
    """
    cfg = replay_config(overrides)
    inspector = VideoInspector()
    api_key = inspector._api_key()
    model = model or get_setting("INSPECTOR_MODEL", None) or "qwen3.6-flash"

    # Gather labeled frames that have a stored image.
    frames = []
    for row in corpus.load_manifest():
        sha = row.get("frame_sha256")
        if not sha:
            continue
        fpath = corpus.frame_path_for(sha)
        if not fpath:
            continue
        frames.append({
            "id": row["id"],
            "file_path": fpath,
            "timestamp_str": row.get("timestamp") or "00:00:00",
            "seconds": float(row.get("seconds") or 0.0),
            "_label": row.get("label"),
        })
    if not frames:
        return {"mode": "frames", "model": model, "error": "no labeled frames with stored images", "frames": 0}

    kept_ids: set[str] = set()
    verdicts: dict[str, dict] = {}
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        detections = inspector.inspect_batch(
            batch, api_key, model, is_coarse=is_coarse, prompt_override=prompt_override,
        )
        by_fid = {d.get("frame_id"): d for d in detections if isinstance(d, dict)}
        for fr in batch:
            det = by_fid.get(fr["id"])
            if det is None:
                continue
            verdicts[fr["id"]] = det
            if is_coarse:
                keep = keep_coarse_detection(det, extreme_recall=cfg["extreme_recall"], hybrid=True, is_two_pass=True,
                                             exclude_male_subject=cfg["exclude_male_subject"], drop_weak_coarse=cfg.get("drop_weak_coarse", False))
            else:
                keep = keep_fine_detection(det, extreme_recall=cfg["extreme_recall"],
                                           require_concrete_evidence=cfg["require_concrete_evidence"],
                                           keep_low_fine=cfg["keep_low_fine"],
                                           exclude_male_subject=cfg["exclude_male_subject"])
            if keep:
                kept_ids.add(fr["id"])

    tp = sum(1 for f in frames if f["_label"] != corpus.NEGATIVE and f["id"] in kept_ids)
    fn = sum(1 for f in frames if f["_label"] != corpus.NEGATIVE and f["id"] not in kept_ids)
    fp = sum(1 for f in frames if f["_label"] == corpus.NEGATIVE and f["id"] in kept_ids)
    tn = sum(1 for f in frames if f["_label"] == corpus.NEGATIVE and f["id"] not in kept_ids)
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * recall * precision / (recall + precision)) if (recall and precision and (recall + precision)) else None
    return {
        "mode": "frames",
        "model": model,
        "is_coarse": is_coarse,
        "prompt_override": bool(prompt_override),
        "frames": len(frames),
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "recall": recall, "precision": precision, "f1": f1,
        "config": cfg,
    }
