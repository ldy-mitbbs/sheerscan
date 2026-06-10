"""Subprocess worker for crop-zoom pose crops.

Run as ``python -m sheerscan.crop_zoom_worker <spec.json>``. Reads a spec, generates
a leg/foot crop for each item, and writes a results JSON.

Why a subprocess: pose detection needs ``ultralytics``, which imports ``opencv``,
whose bundled libavdevice clashes with the inspector's ``PyAV`` in the same
process (``objc Class ... implemented in both`` → possible crashes in the
long-running serve). Isolating pose here keeps opencv out of the serve process
entirely. YOLO loads once and crops every item in this one invocation.

Spec JSON::

    {"video_path": "...", "max_w": 1600, "min_w": 512, "result_path": "...",
     "items": [{"seconds": 12.5, "out_path": "/job/det_001_crop.jpg"}, ...]}

An item may also carry ``"image_path"`` (an already-extracted frame): then the
pose gate runs on that image first and the native frame is decoded only on a
pose hit (the second-chance recall pass) — misses return ``{"skipped": "no_pose"}``.

Result JSON: a list parallel to ``items``, each ``{"used_pose": bool}`` if the
crop was saved, or ``null`` if that item failed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m sheerscan.crop_zoom_worker <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    from . import crop_zoom  # lazily pulls ultralytics/opencv — isolated to this process

    video = spec["video_path"]
    max_w = int(spec.get("max_w", 1600))
    min_w = int(spec.get("min_w", 512))
    results = []
    for it in spec.get("items", []):
        try:
            if it.get("image_path"):
                info = crop_zoom.second_chance_crop(video, float(it["seconds"]), it["image_path"],
                                                    Path(it["out_path"]), max_w=max_w, min_w=min_w)
            else:
                info = crop_zoom.make_crop(video, float(it["seconds"]), Path(it["out_path"]),
                                           max_w=max_w, min_w=min_w)
        except Exception as e:  # never let one bad item abort the batch
            print(f"crop_zoom_worker: item {it.get('seconds')}s failed: {e}", file=sys.stderr)
            info = None
        results.append(info)

    Path(spec["result_path"]).write_text(json.dumps(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
