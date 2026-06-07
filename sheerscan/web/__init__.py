"""Mountable Flask blueprint for the inspector + corpus UI.

``create_inspector_blueprint`` returns a Blueprint with the portable routes
(``/inspect``, ``/corpus``, the ``/api/inspect/*`` job lifecycle, and the
``/api/corpus/*`` labeling API). It takes its dependencies by injection rather
than reaching into a host app's globals:

    bp = create_inspector_blueprint(job_manager=..., cache=...)
    app.register_blueprint(bp)

* ``job_manager`` — a :class:`sheerscan.inspector.VideoInspectorJobManager`.
* ``cache`` — optional. If it exposes ``list_inspector_eval_marks`` /
  ``put_inspector_eval_run`` (as a host app's Cache may) those power the eval
  endpoints; otherwise eval falls back to the file-based manual marks and run
  persistence is skipped. Pass ``None`` to run purely file-based.
* ``static_dir`` — defaults to this package's bundled HTML.

Routes that need a host-specific media catalog or player integration (manual-mark
resolution, media file listing) are intentionally NOT here — they stay in the
host app.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, request, send_from_directory

from ..runtime import get_setting, get_local_video_dir
from ..inspector import evaluate_inspector_result, list_manual_marks
from .. import corpus as corpus_mod

_DEFAULT_STATIC = Path(__file__).resolve().parent / "static"


def _eval_marks(cache, container_path):
    """Prefer a host cache's eval marks; fall back to file-based manual marks."""
    if cache is not None and hasattr(cache, "list_inspector_eval_marks"):
        try:
            marks = cache.list_inspector_eval_marks(container_path)
            if marks:
                return marks
        except Exception:
            pass
    return list_manual_marks(container_path)


def create_inspector_blueprint(
    *,
    job_manager,
    cache=None,
    static_dir: Optional[Path] = None,
    url_prefix: str = "",
    name: str = "sheerscan",
) -> Blueprint:
    static_dir = Path(static_dir) if static_dir else _DEFAULT_STATIC
    bp = Blueprint(name, __name__, url_prefix=url_prefix or "")

    def _page(filename):
        resp = send_from_directory(str(static_dir), filename)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    # ------------------------------------------------------------- inspect
    @bp.post("/api/inspect/jobs")
    def api_inspect_start_job():
        data = request.get_json(force=True, silent=True) or {}
        container_path = (data.get("container_path") or "").strip()
        if not container_path:
            return jsonify({"error": "missing container_path"}), 400

        interval_raw = data.get("interval")
        try:
            interval = int(interval_raw)
            if interval <= 0:
                interval = int(get_setting("INSPECTOR_INTERVAL", 5))
        except (TypeError, ValueError):
            interval = int(get_setting("INSPECTOR_INTERVAL", 5))

        try:
            job_id = job_manager.start_job(container_path, interval)
            return jsonify({"ok": True, "job_id": job_id})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.get("/api/inspect/jobs")
    def api_inspect_list_jobs():
        return jsonify(job_manager.list_jobs())

    @bp.get("/api/inspect/jobs/<job_id>")
    def api_inspect_get_job(job_id):
        job = job_manager.get_job(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        if job.get("result"):
            try:
                marks = _eval_marks(cache, job["result"].get("container_path"))
                job["result"]["eval"] = evaluate_inspector_result(job["result"], marks)
            except Exception:
                pass
        return jsonify(job)

    @bp.get("/api/inspect/jobs/<job_id>/eval")
    def api_inspect_eval_job(job_id):
        job = job_manager.get_job(job_id)
        if not job or not job.get("result"):
            return jsonify({"error": "job result not found"}), 404
        try:
            tolerance = float(request.args.get("tolerance", 30.0))
            merge_window = float(request.args.get("merge_window", 8.0))
        except ValueError:
            return jsonify({"error": "invalid tolerance or merge_window"}), 400
        marks = _eval_marks(cache, job["result"].get("container_path"))
        metrics = evaluate_inspector_result(
            job["result"],
            marks,
            tolerance_seconds=tolerance,
            merge_window_seconds=merge_window,
        )
        saved = None
        if cache is not None and hasattr(cache, "put_inspector_eval_run"):
            try:
                saved = cache.put_inspector_eval_run({
                    "job_id": job_id,
                    "container_path": job["result"].get("container_path"),
                    "model": job["result"].get("model"),
                    "fine_model": job["result"].get("fine_model"),
                    "verify_model": job["result"].get("verify_model"),
                    "tolerance_seconds": tolerance,
                    "merge_window_seconds": merge_window,
                    "metrics": metrics,
                })
            except Exception:
                saved = None
        return jsonify({"ok": True, "eval": metrics, "run": saved})

    @bp.get("/api/inspect/artifact/<job_id>/<path:filename>")
    def api_inspect_get_artifact(job_id, filename):
        if ".." in job_id or ".." in filename or filename.startswith("/"):
            return jsonify({"error": "invalid path"}), 400
        job_dir = get_local_video_dir() / "inspections" / job_id
        if not job_dir.exists() or not job_dir.is_dir():
            return jsonify({"error": "job directory not found"}), 404
        return send_from_directory(str(job_dir), filename)

    @bp.get("/api/inspect/frame/<job_id>/<filename>")
    def api_inspect_get_frame(job_id, filename):
        return api_inspect_get_artifact(job_id, filename)

    @bp.get("/api/inspect/manual-mark/screenshot/<filename>")
    def api_inspect_manual_mark_screenshot(filename):
        if ".." in filename or "/" in filename:
            return jsonify({"error": "invalid path"}), 400
        marks_dir = get_local_video_dir() / "inspections" / "manual_marks"
        if not marks_dir.exists():
            return jsonify({"error": "manual mark screenshot directory not found"}), 404
        return send_from_directory(str(marks_dir), filename)

    @bp.post("/api/inspect/jobs/<job_id>/feedback")
    def api_inspect_feedback(job_id):
        if ".." in job_id:
            return jsonify({"error": "invalid path"}), 400
        data = request.get_json(force=True, silent=True) or {}
        label = (data.get("label") or "").strip()
        allowed_labels = {"false_positive", "ok", "like", "love", "unsure"}
        if label not in allowed_labels:
            return jsonify({"error": "invalid label"}), 400
        try:
            det_index = int(data.get("detection_index"))
        except (TypeError, ValueError):
            return jsonify({"error": "invalid detection_index"}), 400

        inspections_dir = get_local_video_dir() / "inspections"
        job_dir = inspections_dir / job_id
        result_path = job_dir / "result.json"
        if not result_path.exists():
            return jsonify({"error": "result not found"}), 404
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            return jsonify({"error": f"failed to read result: {e}"}), 500

        detections = result.get("visual_detections") or []
        if det_index < 0 or det_index >= len(detections):
            return jsonify({"error": "detection_index out of range"}), 400

        feedback = {
            "label": label,
            "note": (data.get("note") or "").strip(),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        detections[det_index]["feedback"] = feedback
        result["visual_detections"] = detections
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        record = {
            "job_id": job_id,
            "detection_index": det_index,
            "container_path": result.get("container_path"),
            "video_file": result.get("video_file"),
            "detection": detections[det_index],
            "feedback": feedback,
        }
        feedback_path = inspections_dir / "feedback.jsonl"
        with feedback_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        job = job_manager.get_job(job_id)
        if job and isinstance(job.get("result"), dict):
            job["result"] = result
        return jsonify({"ok": True, "feedback": feedback})

    @bp.get("/inspect")
    def page_inspect():
        return _page("inspect.html")

    # -------------------------------------------------------------- corpus
    @bp.get("/api/corpus")
    def api_corpus_list():
        label = request.args.get("label")
        rows = corpus_mod.load_manifest()
        if label in ("positive", "negative"):
            rows = [r for r in rows if (r.get("label") or "positive") == label]
        rows.sort(key=lambda r: (r.get("video_id") or "", float(r.get("seconds") or 0.0)))
        for r in rows:
            r["frame_url"] = (
                f"/api/corpus/frame/{r['frame_sha256']}" if r.get("frame_sha256") else None
            )
        return jsonify({"examples": rows, "stats": corpus_mod.corpus_stats()})

    @bp.post("/api/corpus/harvest")
    def api_corpus_harvest():
        stats = corpus_mod.harvest()
        return jsonify({"ok": True, "stats": stats, "corpus": corpus_mod.corpus_stats()})

    @bp.post("/api/corpus/label")
    def api_corpus_label():
        data = request.get_json(force=True, silent=True) or {}
        example_id = (data.get("id") or "").strip()
        label = (data.get("label") or "").strip().lower()
        if not example_id or label not in ("positive", "negative"):
            return jsonify({"ok": False, "error": "id and label (positive|negative) required"}), 400
        try:
            updated = corpus_mod.set_label(example_id, label)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if not updated:
            return jsonify({"ok": False, "error": "example not found"}), 404
        return jsonify({"ok": True, "example": updated})

    @bp.get("/api/corpus/frame/<sha>")
    def api_corpus_frame(sha):
        if not re.fullmatch(r"[0-9a-f]{16,64}", sha or ""):
            return jsonify({"error": "invalid hash"}), 400
        path = corpus_mod.frame_path_for(sha)
        if not path or not path.exists():
            return jsonify({"error": "frame not found"}), 404
        return send_from_directory(str(path.parent), path.name)

    @bp.get("/corpus")
    def page_corpus():
        return _page("corpus.html")

    return bp
