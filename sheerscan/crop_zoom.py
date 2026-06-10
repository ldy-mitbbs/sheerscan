"""Crop-zoom: pose-localize the leg/foot region and crop it from the native-res
frame, so a VLM (and the human reviewer) gets real leg detail instead of a
downscaled thumbnail.

Why this exists: the dominant inspector false positive is "smooth bare legs read
as 肉色丝袜". Extensive A/B testing (scripts/montage_ab.py, see the
montage-motion memory / TECHNICAL_INSPECTOR notes) showed the stocking cue is NOT
fundamentally unreadable on a single frame — it is destroyed by **downscaling the
leg region to a few pixels**. A tight, pose-localized crop reliably recovers
clearly-stockinged cases two independent models miss at full-frame. Motion
(montage) does NOT help — the hard cases are static. So the lever is
resolution/localization, and this module is that lever.

Pure helpers, no dependency on video_inspector (avoids an import cycle): the
caller does any VLM re-scoring with its own inspect_batch. ``ultralytics`` is
lazily imported and optional — without it we fall back to a fixed heuristic
region (looser, but still a zoom).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# COCO-17 leg keypoints (hips/knees/ankles) and the knee/ankle subset used to
# decide whether a person's legs are actually visible.
_LEG_KP = [11, 12, 13, 14, 15, 16]
_KNEE_ANKLE_KP = [13, 14, 15, 16]

_POSE_MODEL = None
_POSE_TRIED = False

# Crop-aware re-score prompt: matched to the live coarse task but tells the model
# this is a zoomed leg/foot region and asks for a graded 0-1 score. Callers feed
# it to inspect_batch as a prompt_override and parse the `score` field.
CROP_RESCORE_PROMPT = (
    "你是丝袜识别视觉模型。下面是从原视频画面中**裁剪并放大的腿/脚部区域**"
    "（分辨率比全帧高，细节更清楚）。目标只有一种：**薄款、半透明、肉色/肤色或浅灰色的丝袜/连裤袜**"
    "——它薄到能透出腿的肤色和轮廓，只是给皮肤加了一层均匀光泽。\n\n"
    "按这个顺序判断：\n"
    "1. **先看是不是不透明衣物**。如果这块区域是**深色或不透明的布料**——深色/彩色长裤、短裤、裙子、"
    "厚打底裤/leggings、黑丝、彩色丝袜——它**遮住了腿、透不出肤色**，那就**不是目标**，score≤0.1。"
    "注意：深灰色、深褐色、不透明＝衣物，不要因为‘均匀’或‘有光泽’就当成丝袜。\n"
    "2. 如果能看到**腿/脚的皮肤本身**，再判断是裸的还是covered：\n"
    "   - 裸腿/裸脚：清晰皮肤纹理、自然关节褶皱、趾甲高光、肤色不均 → score 低（0.1~0.3）；\n"
    "   - 薄丝袜：在仍能透出肤色的前提下，有连续均匀的薄层光泽、袜口/脚尖缝线、趾甲与脚趾缝被柔化"
    "→ score 高（0.7~1）。\n"
    "**关键**：score 高只给‘能透出腿、加了一层薄光泽’的情况；只要是不透明、遮住腿的布料，无论什么颜色都给低分。\n\n"
    "只返回严格 JSON，detections 数组里**必须恰好有一个对象**，frame_id 用给定的 Frame ID：\n"
    '{ "detections": [ { "frame_id": "<给定ID>", "score": <0到1之间的小数>, '
    '"reason": "中文：放大区域是皮肤还是不透明衣物，看到的具体证据" } ] }\n'
    "score = 出现薄款半透明肉色/浅灰色丝袜的概率（0=裸腿**或**不透明衣物，1=肯定是薄丝袜）。必须给 0~1 连续值，不要弃权。"
)


def pose_available() -> bool:
    return _load_pose_model() is not None


def _load_pose_model():
    """Lazily load YOLO-pose; cache the (possibly-None) result. Returns None if
    ultralytics/the weights aren't available, so callers degrade gracefully."""
    global _POSE_MODEL, _POSE_TRIED
    if _POSE_TRIED:
        return _POSE_MODEL
    _POSE_TRIED = True
    try:
        from ultralytics import YOLO
        _POSE_MODEL = YOLO("yolo11n-pose.pt")
    except Exception as e:  # ultralytics missing, weights undownloadable, etc.
        print(f"crop_zoom: pose model unavailable ({str(e)[:100]}); using heuristic crop")
        _POSE_MODEL = None
    return _POSE_MODEL


def extract_native_frame(video_path, seconds: float, max_w: int = 1600,
                         pre: float = 1.0) -> Image.Image | None:
    """Extract one frame at ~`seconds` at up to `max_w` wide (native-ish res).

    Two-stage seek (fast input seek + short decoded output seek) plus corrupt-
    packet tolerance, matching the inspector's handling of damaged .ts GOPs."""
    input_seek = max(0.0, float(seconds) - pre)
    out_seek = float(seconds) - input_seek
    with tempfile.TemporaryDirectory(prefix="cropzoom-") as td:
        out = str(Path(td) / "f.jpg")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
            "-ss", f"{input_seek:.3f}", "-i", str(video_path), "-ss", f"{out_seek:.3f}",
            # yadif deinterlaces the interlaced MPEG2 .ts source — without it the
            # crop shows comb/scanline artifacts that wreck fine leg detail.
            "-vf", f"yadif,scale='min({max_w},iw)':-2", "-frames:v", "1", "-q:v", "2", out,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0 or not Path(out).exists():
            return None
        return Image.open(out).convert("RGB").copy()


_HEURISTIC_BOX = (0.08, 0.40, 0.92, 1.0)  # normalized (x0,y0,x1,y1)


def heuristic_crop_region(img: Image.Image):
    """Fallback when pose is unavailable: lower 60% of the frame, central 84%.
    Returns (crop, normalized_box)."""
    w, h = img.size
    fx0, fy0, fx1, fy1 = _HEURISTIC_BOX
    crop = img.crop((int(w * fx0), int(h * fy0), int(w * fx1), int(h * fy1)))
    return crop, _HEURISTIC_BOX


def pose_crop_region(img: Image.Image, min_w: int = 512):
    """Tight crop around the most prominent person's legs/feet via YOLO-pose.

    Returns (crop, normalized_box) or None (caller falls back to heuristic) when
    no person with visible knee/ankle keypoints is found."""
    model = _load_pose_model()
    if model is None:
        return None
    try:
        res = model(np.asarray(img), verbose=False)[0]
    except Exception as e:
        print(f"crop_zoom: pose inference failed ({str(e)[:100]})")
        return None
    if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
        return None
    kps = res.keypoints.data.cpu().numpy()              # (n_person, 17, 3): x,y,conf
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else None

    best, best_area = None, -1.0
    for i, person in enumerate(kps):
        legs = person[_KNEE_ANKLE_KP]
        if len(legs[legs[:, 2] > 0.30]) < 2:            # need >=2 visible knee/ankle pts
            continue
        area = float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])) if boxes is not None else 1.0
        if area > best_area:                            # most prominent subject wins
            best, best_area = person, area
    if best is None:
        return None

    pts = best[_LEG_KP]
    pts = pts[pts[:, 2] > 0.30]
    if len(pts) < 2:
        return None
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1, y0, y1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    W, H = img.size
    box = (max(0, int(x0 - max(bw * 0.45, W * 0.04))),
           max(0, int(y0 - bh * 0.15)),
           min(W, int(x1 + max(bw * 0.45, W * 0.04))),
           min(H, int(y1 + max(bh * 0.55, H * 0.07))))   # extend down for the feet
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    crop = img.crop(box)
    if crop.width < min_w:
        crop = crop.resize((min_w, max(1, int(crop.height * min_w / crop.width))), Image.LANCZOS)
    norm_box = (box[0] / W, box[1] / H, box[2] / W, box[3] / H)
    return crop, norm_box


def make_crop(video_path, seconds: float, out_path: Path, *, max_w: int = 1600,
              min_w: int = 512, prefer_pose: bool = True) -> dict | None:
    """Extract the native frame at `seconds`, crop the leg/foot region, save to
    `out_path`. Returns {"used_pose": bool} or None if extraction failed.

    Calls pose detection (opencv) — run this in the subprocess worker, not the
    serve process. For an opencv-free path use ``make_crop_heuristic``."""
    frame = extract_native_frame(video_path, seconds, max_w=max_w)
    if frame is None:
        return None
    res = pose_crop_region(frame, min_w=min_w) if prefer_pose else None
    used_pose = res is not None
    if res is None:
        res = heuristic_crop_region(frame)
    region, box = res
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region.save(out_path, "JPEG", quality=92)
    return {"used_pose": used_pose, "box": [round(v, 4) for v in box]}


def second_chance_crop(video_path, seconds: float, image_path, out_path, *,
                       max_w: int = 1600, min_w: int = 512) -> dict | None:
    """Pose-gate + native-res leg crop for the second-chance recall pass.

    Pose runs on the already-extracted coarse frame (``image_path``) so frames
    without detectable legs cost no video decode at all; only on a pose hit is
    the native frame extracted and the (scaled) leg box cropped from it.
    Returns ``{"used_pose": True, "box": [...]}`` when a crop was saved,
    ``{"skipped": "no_pose"}`` when no legs were localized, ``None`` on failure.
    Calls pose detection (opencv) — run in the subprocess worker only."""
    try:
        with Image.open(image_path) as im:
            res = pose_crop_region(im.convert("RGB"), min_w=min_w)
    except Exception:
        return None
    if res is None:
        return {"skipped": "no_pose"}
    _, box = res
    frame = extract_native_frame(video_path, seconds, max_w=max_w)
    if frame is None:
        return None
    W, H = frame.size
    crop = frame.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)))
    if crop.width < min_w:
        crop = crop.resize((min_w, max(1, int(crop.height * min_w / crop.width))), Image.LANCZOS)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, "JPEG", quality=92)
    return {"used_pose": True, "box": [round(v, 4) for v in box]}


def make_crop_heuristic(video_path, seconds: float, out_path: Path, *, max_w: int = 1600) -> dict | None:
    """Opencv-free crop (no pose): extract + fixed lower-frame region. Safe to
    call in the serve process; used as the fallback when the pose subprocess is
    unavailable."""
    frame = extract_native_frame(video_path, seconds, max_w=max_w)
    if frame is None:
        return None
    region, box = heuristic_crop_region(frame)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region.save(out_path, "JPEG", quality=92)
    return {"used_pose": False, "box": [round(v, 4) for v in box]}


def make_crops_via_subprocess(video_path, items: list[dict], *, max_w: int = 1600,
                              min_w: int = 512, timeout: float = 600.0) -> list | None:
    """Generate pose crops for many items in ONE isolated subprocess, so
    ultralytics/opencv never load in the caller (PyAV) process.

    `items`: [{"seconds": float, "out_path": str}, ...]. Returns a list parallel
    to `items` ({"used_pose": bool}|None per item), or None if the whole worker
    failed (caller should fall back to make_crop_heuristic)."""
    if not items:
        return []
    with tempfile.TemporaryDirectory(prefix="cropzoom-spec-") as td:
        spec_path = Path(td) / "spec.json"
        result_path = Path(td) / "result.json"
        spec_path.write_text(json.dumps({
            "video_path": str(video_path), "max_w": max_w, "min_w": min_w,
            "result_path": str(result_path),
            "items": [{"seconds": float(i["seconds"]), "out_path": str(i["out_path"]),
                       **({"image_path": str(i["image_path"])} if i.get("image_path") else {})}
                      for i in items],
        }), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "sheerscan.crop_zoom_worker", str(spec_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        except Exception as e:
            print(f"crop_zoom worker did not run: {e}")
            return None
        if proc.returncode != 0 or not result_path.exists():
            print(f"crop_zoom worker failed (rc={proc.returncode}): {(proc.stderr or '')[-300:]}")
            return None
        try:
            out = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"crop_zoom worker bad output: {e}")
            return None
        return out if isinstance(out, list) and len(out) == len(items) else None
