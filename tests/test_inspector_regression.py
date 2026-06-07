"""Regression tests for the inspector replay harness.

Two layers:

* ``test_postprocess_replay_fixture`` — fully deterministic, uses a synthetic
  trace + corpus. Runs everywhere (no API, no local data) and pins the exact
  reconstruction + scoring behaviour.

* ``test_corpus_replay_meets_baseline`` — replays this machine's real traces
  against the committed ``baseline.json`` and asserts recall/precision haven't
  regressed. Skips cleanly when local traces or a baseline are absent (CI / a
  fresh clone), since the sensitive frames/traces are never committed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sheerscan.replay import (
    replay_trace_postprocess,
    replay_config,
    replay_postprocess_over_corpus,
)
from sheerscan.inspector import evaluate_against_corpus
from sheerscan import corpus


def _cfg(**over):
    base = {
        "extreme_recall": False,
        "require_concrete_evidence": False,
        "keep_low_fine": True,
        "keep_coarse_on_fine_empty": True,
        "exclude_male_subject": True,
        "rep_offset": 4.0,
        "dedup_window": 3.0,
        "min_scene_support": 1,
        "scene_support_window": 10.0,
    }
    base.update(over)
    return base


# A synthetic hybrid trace: a coarse pass that seeds candidates, then a video
# fine pass that confirms one real hit and emits one false-positive claim.
FIXTURE_TRACE = [
    {
        "stage": "coarse_screenshot",
        "frames": [
            {"frame_id": "f1", "seconds": 100.0},
            {"frame_id": "f2", "seconds": 300.0},
        ],
        "parsed_detections": [
            {"frame_id": "f1", "confidence": "medium", "reason": "脚背疑似肉色丝袜覆盖"},
            {"frame_id": "f2", "confidence": "low", "reason": "中景，女性穿着裤子，没有看到丝袜"},  # false-positive language
        ],
    },
    {
        "stage": "fine_video",
        "start_seconds": 95.0,
        "end_seconds": 125.0,
        "duration": 30.0,
        "parsed_detections": [
            {"time": "00:05", "confidence": "high", "reason": "脚尖缝线清晰，半透明织物覆盖脚背"},
        ],
    },
]


def test_postprocess_replay_fixture():
    # default config: false-positive coarse candidate dropped; fine hit kept
    dets = replay_trace_postprocess(FIXTURE_TRACE, _cfg())
    assert len(dets) == 1
    assert dets[0]["confidence"] == "high"
    assert dets[0]["seconds"] == 104.0  # 95 start + (00:05 + 4s representative offset)

    # score against a corpus: one positive at 100s, one negative at 300s
    corpus_by_video = {
        "vid1": {
            "positives": [{"id": "p1", "seconds": 100.0}],
            "negatives": [{"id": "n1", "seconds": 300.0}],
            "container_path": "/data/x.ts",
        }
    }
    metrics = evaluate_against_corpus({"vid1": dets}, corpus_by_video, tolerance_seconds=10.0)
    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["false_positives"] == 0  # the FP-language candidate was filtered out
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_postprocess_replay_coarse_fallback_when_fine_empty():
    trace = [
        {
            "stage": "coarse_screenshot",
            "frames": [{"frame_id": "f1", "seconds": 100.0}],
            "parsed_detections": [{"frame_id": "f1", "confidence": "medium", "reason": "脚背疑似丝袜"}],
        },
        {"stage": "fine_video", "start_seconds": 95.0, "end_seconds": 125.0, "duration": 30.0, "parsed_detections": []},
    ]
    # fine ran but kept nothing -> coarse candidate preserved for review
    kept = replay_trace_postprocess(trace, _cfg(keep_coarse_on_fine_empty=True))
    assert len(kept) == 1
    assert kept[0].get("fine_empty_fallback") is True
    # ...unless the fallback is disabled
    assert replay_trace_postprocess(trace, _cfg(keep_coarse_on_fine_empty=False)) == []


def test_postprocess_replay_strict_evidence_override_drops_weak_fine_hit():
    trace = [
        {
            "stage": "fine_video", "start_seconds": 0.0, "end_seconds": 30.0, "duration": 30.0,
            "parsed_detections": [
                {"time": "00:05", "confidence": "high", "reason": "腿部光滑均匀，符合肉色丝袜特征"},  # weak, no concrete evidence
            ],
        },
    ]
    assert len(replay_trace_postprocess(trace, _cfg(require_concrete_evidence=False))) == 1
    assert replay_trace_postprocess(trace, _cfg(require_concrete_evidence=True)) == []


# ---------------------------------------------------------------- real-data baseline

def _has_local_traces() -> bool:
    root = corpus.get_local_video_dir() / "inspections"
    return root.exists() and any(p.joinpath("trace.json").exists() for p in root.iterdir() if p.is_dir())


@pytest.mark.skipif(not corpus.baseline_path().exists(), reason="no committed baseline")
@pytest.mark.skipif(not corpus.manifest_path().exists(), reason="no corpus manifest")
def test_corpus_replay_meets_baseline():
    if not _has_local_traces():
        pytest.skip("no local inspector traces on this machine")

    baseline = json.loads(corpus.baseline_path().read_text(encoding="utf-8"))
    metrics = replay_postprocess_over_corpus(
        tolerance_seconds=baseline.get("tolerance_seconds", 30.0),
        merge_window_seconds=baseline.get("merge_window_seconds", 8.0),
    )
    eps = 1e-6
    for key in ("recall", "precision", "f1"):
        base, cur = baseline.get(key), metrics.get(key)
        if base is None or cur is None:
            continue
        assert cur >= base - eps, (
            f"inspector {key} regressed: {cur:.4f} < baseline {base:.4f}. "
            f"If this drop is intentional, re-record with "
            f"`sheerscan replay --mode postprocess --update-baseline`."
        )
