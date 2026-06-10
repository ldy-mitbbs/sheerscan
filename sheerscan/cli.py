"""Standalone CLI: ``sheerscan <command>``.

    sheerscan inspect <video>            run the pipeline on one file
    sheerscan corpus harvest|stats|list  manage the regression corpus
    sheerscan replay --mode postprocess  free Tier-A replay over the corpus
    sheerscan replay --mode frames ...    Tier-B model/prompt A/B (costs API)

The host app keeps its own CLI and calls the library directly; this is
for using sheerscan on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _parse_overrides(pairs) -> dict:
    """Parse --set key=value overrides for replay post-processing knobs."""
    overrides: dict = {}
    bool_keys = {"extreme_recall", "require_concrete_evidence", "keep_low_fine",
                 "keep_coarse_on_fine_empty", "exclude_male_subject", "skip_fine", "drop_weak_coarse",
                 "reason_filter"}
    float_keys = {"rep_offset", "dedup_window", "scene_support_window", "crop_gate"}
    int_keys = {"min_scene_support"}
    for pair in pairs or []:
        if "=" not in pair:
            sys.exit(f"error: --set expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in bool_keys:
            overrides[key] = value.lower() in {"1", "true", "yes", "on"}
        elif key in float_keys:
            overrides[key] = float(value)
        elif key in int_keys:
            overrides[key] = int(value)
        else:
            sys.exit(f"error: unknown override key {key!r}; allowed: {sorted(bool_keys | float_keys | int_keys)}")
    return overrides


def _slim_baseline(metrics: dict) -> dict:
    slim = {k: metrics[k] for k in ("recall", "precision", "f1", "true_positives", "false_negatives",
                                    "false_positives", "positive_events", "negative_events",
                                    "tolerance_seconds", "merge_window_seconds") if k in metrics}
    per_video = {}
    for vid, m in (metrics.get("per_video") or {}).items():
        per_video[vid] = {
            "recall": m.get("recall"), "precision": m.get("precision"),
            "true_positives": m.get("true_positives"), "false_negatives": m.get("false_negatives"),
            "false_positives": m.get("false_positives"),
            "cases": [{"seconds": c.get("seconds"), "matched": c.get("matched")} for c in (m.get("cases") or [])],
        }
    slim["per_video"] = per_video
    return slim


def cmd_inspect(args) -> int:
    """Run the pipeline on a single video and print the detections."""
    from .inspector import VideoInspectorJobManager
    import time

    jm = VideoInspectorJobManager()
    job_id = jm.start_job(args.video, args.interval)
    print(f"job {job_id} started; polling…")
    while True:
        job = jm.get_job(job_id) or {}
        status = job.get("status")
        if status in ("completed", "failed"):
            break
        msg = job.get("message") or ""
        print(f"  [{job.get('progress', 0)}%] {status}: {msg}", flush=True)
        time.sleep(2.0)

    if job.get("status") == "failed":
        sys.exit(f"error: {job.get('error') or job.get('message')}")
    result = job.get("result") or {}
    dets = result.get("visual_detections") or []
    print(f"\n{len(dets)} detection(s):")
    for d in dets:
        secs = d.get("seconds")
        ts = f"{int(secs // 60):02d}:{int(secs % 60):02d}" if isinstance(secs, (int, float)) else "?"
        print(f"  {ts}  conf={d.get('confidence', '?'):<6} {(d.get('reason') or '')[:80]}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


def cmd_corpus(args) -> int:
    from . import corpus

    if args.action == "harvest":
        stats = corpus.harvest(verbose=False)
        print(
            f"Harvested corpus: {stats['total_examples']} examples "
            f"({stats['marks']} marks + {stats['feedback']} feedback), "
            f"{stats['frames_copied']} frame(s) stored, {stats['videos']} video(s), "
            f"{stats['skipped']} skipped."
        )
        s = corpus.corpus_stats()
        print(f"  positives={s['positives']} negatives={s['negatives']} with_frames={s['with_frames']}")
        print(f"  manifest: {corpus.manifest_path()}")
        print(f"  frames:   {corpus.frames_dir()}")
    elif args.action == "stats":
        print(json.dumps(corpus.corpus_stats(), indent=2))
    elif args.action == "list":
        for row in corpus.load_manifest():
            print(
                f"{row.get('label', '?'):<8} {row.get('kind', '?'):<6} "
                f"{row.get('timestamp', ''):<10} {row.get('video_id', '')} "
                f"{(row.get('reason') or '')[:60]}"
            )
    return 0


def cmd_replay(args) -> int:
    from . import replay, corpus
    from .inspector import compare_eval_runs

    overrides = _parse_overrides(args.set)

    if args.mode == "frames":
        prompt_override = None
        if args.prompt_file:
            prompt_override = Path(args.prompt_file).read_text(encoding="utf-8")
        metrics = replay.replay_frames_classification(
            model=args.model, is_coarse=not args.fine,
            prompt_override=prompt_override, overrides=overrides,
        )
        print(
            f"Frame classification ({metrics.get('model')}, {'fine' if args.fine else 'coarse'}): "
            f"frames={metrics.get('frames')} tp={metrics.get('true_positives')} fn={metrics.get('false_negatives')} "
            f"fp={metrics.get('false_positives')} tn={metrics.get('true_negatives')} "
            f"recall={metrics.get('recall')} precision={metrics.get('precision')} f1={metrics.get('f1')}"
        )
    else:
        metrics = replay.replay_postprocess_over_corpus(
            overrides=overrides, tolerance_seconds=args.tolerance, merge_window_seconds=args.merge_window,
        )
        print(
            f"Post-processing replay over corpus: videos={metrics['videos']} "
            f"(with detections {metrics['videos_with_detections']}, "
            f"uncovered {len(metrics.get('uncovered_videos') or [])}/{metrics.get('corpus_videos_total', metrics['videos'])}) "
            f"tp={metrics['true_positives']} fn={metrics['false_negatives']} fp={metrics['false_positives']} "
            f"recall={metrics['recall']} precision={metrics['precision']} f1={metrics['f1']}"
        )
        rf = metrics.get("reason_filter")
        if rf:
            print(f"  reason filter: available={rf['available']} kept {rf['kept']}/{rf['input']} (dropped {rf['dropped']})")
        if args.compare:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
            diff = compare_eval_runs(baseline, metrics)
            print(
                f"  vs baseline: Δrecall={diff['recall_delta']} Δprecision={diff['precision_delta']} "
                f"Δf1={diff['f1_delta']} fixed={len(diff['fixed_events'])} regressed={len(diff['regressed_events'])} "
                f"{'*** REGRESSION ***' if diff['is_regression'] else 'ok'}"
            )

    if args.out:
        Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    if args.update_baseline and args.mode == "postprocess":
        corpus.baseline_path().parent.mkdir(parents=True, exist_ok=True)
        corpus.baseline_path().write_text(
            json.dumps(_slim_baseline(metrics), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"Updated baseline: {corpus.baseline_path()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sheerscan", description="Hosiery-detection video inspector.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect", help="Run the pipeline on one video file.")
    sp.add_argument("video", help="path to the video file")
    sp.add_argument("--interval", type=int, default=5, help="frame sampling interval (seconds)")
    sp.add_argument("--json", help="write the full result JSON to this path")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("corpus", help="Manage the durable regression corpus.")
    sp.add_argument("action", choices=["harvest", "stats", "list"])
    sp.set_defaults(func=cmd_corpus)

    sp = sub.add_parser("replay", help="Replay the corpus offline (Tier A) or re-classify frames (Tier B).")
    sp.add_argument("--mode", choices=["postprocess", "frames"], default="postprocess")
    sp.add_argument("--model", help="model id for --mode frames")
    sp.add_argument("--fine", action="store_true", help="use the fine prompt for --mode frames")
    sp.add_argument("--prompt-file", help="swap in a candidate prompt for --mode frames")
    sp.add_argument("--set", action="append", help="override a knob, e.g. --set dedup_window=3.0")
    sp.add_argument("--tolerance", type=float, default=30.0)
    sp.add_argument("--merge-window", dest="merge_window", type=float, default=8.0)
    sp.add_argument("--compare", help="baseline.json to diff against")
    sp.add_argument("--out", help="write metrics JSON to this path")
    sp.add_argument("--update-baseline", action="store_true", help="re-record the committed baseline")
    sp.set_defaults(func=cmd_replay)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
