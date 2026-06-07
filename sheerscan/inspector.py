"""Core inspector: frame extraction + VLM classification + post-processing.

READING THIS FILE — what's live vs. optional:

The **recommended pipeline** (what a standalone install runs by default, and the
product answer the README defends) is small and lives in:
  • ``VideoInspector.extract_frames``      — single ffmpeg keyframe pass
  • ``VideoInspector.inspect_batch``       — coarse VLM call + simple prompt
  • ``VideoInspector.run_inspection``      — the coarse-only orchestration path
  • ``keep_coarse_detection`` / ``finalize_detections`` — post-processing
  • the semantic reason filter (in ``reason_filter.py``), the sole judge
  • ``VideoInspectorJobManager``           — async job lifecycle

Everything else is **opt-in / legacy, OFF by default**, kept for experiments and
A/B history — it is gated behind env flags and does NOT run in the default
pipeline. Treat these as feature flags, not dead code:
  • fine video pass        — ``_run_hybrid_video_fine_pass`` (INSPECTOR_SKIP_FINE_PASS=1 skips it; default skipped standalone). Hurt recall+precision; several of its pure helpers are retained because the regression replay reconstructs through them.
  • native_video mode      — ``_run_native_video_inspection`` (INSPECTOR_MODE)
  • screenshot two-pass    — the non-default ``INSPECTOR_MODE=screenshot`` branch
  • verifier pass          — ``verify_visual_detections`` (INSPECTOR_VERIFY_ENABLED)
  • local CLIP prefilter   — ``_clip_prefilter_frames`` etc. (INSPECTOR_LOCAL_CLIP_ENABLED)
  • crop-zoom enrichment   — ``_apply_crop_zoom`` (INSPECTOR_CROP_ZOOM)
  • extreme-recall / keyword / confidence-drop / strict-evidence branches — all default off
"""
import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import requests
import av
from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS

from .runtime import (
    get_secret,
    get_setting,
    get_local_video_dir,
    to_host_path,
    to_container_path,
)

_LOCAL_CLIP_STATE = {
    "available": None,
    "error": "",
    "model_name": "",
    "device": "cpu",
    "model": None,
    "processor": None,
    "torch": None,
}

INSPECTOR_MODEL_OPTIONS = [
    {
        "id": "google/gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
        "note": "准确率/速度/成本比较均衡，适合粗检",
    },
    {
        "id": "google/gemini-2.5-pro",
        "label": "Gemini 2.5 Pro",
        "note": "更强视觉判断，适合精检；成本较高但更准",
    },
    {
        "id": "google/gemini-2.0-flash-lite-001",
        "label": "Gemini 2.0 Flash Lite",
        "note": "稳定、便宜，适合大量粗扫但准确率较保守",
    },
    {
        "id": "google/gemini-2.5-flash-lite",
        "label": "Gemini 2.5 Flash Lite",
        "note": "低成本视觉模型，适合大量视频",
    },
    {
        "id": "openai/gpt-4o-mini",
        "label": "GPT-4o mini",
        "note": "OpenAI 视觉模型，适合作为 Gemini/Qwen 的备选",
    },
    {
        "id": "openai/gpt-4.1-mini",
        "label": "GPT-4.1 mini",
        "note": "OpenAI 视觉模型，稳定性通常较好",
    },
    {
        "id": "openai/gpt-5",
        "label": "GPT-5",
        "note": "更强但更贵，适合最终复核疑难截图",
    },
    {
        "id": "anthropic/claude-3.5-haiku",
        "label": "Claude 3.5 Haiku",
        "note": "Anthropic 视觉模型，适合替代供应商限流",
    },
    {
        "id": "qwen/qwen3.5-plus-20260420",
        "label": "Qwen3.5 Plus",
        "note": "Qwen 高质量视觉/视频模型",
    },
    {
        "id": "qwen/qwen3.6-flash",
        "label": "Qwen3.6 Flash",
        "note": "Qwen 低成本视觉/视频模型；当前实测可用，适合粗检",
    },
    {
        "id": "qwen3.6-flash",
        "label": "MuleRouter Qwen3.6 Flash",
        "note": "MuleRouter 官方 Qwen ID；适合粗检",
    },
    {
        "id": "qwen3-vl-flash",
        "label": "MuleRouter Qwen3 VL Flash",
        "note": "极低成本视觉语言模型；只支持图像输入，适合截图粗检",
    },
    {
        "id": "qwen3.5-omni-flash",
        "label": "MuleRouter Qwen3.5 Omni Flash",
        "note": "低成本多模态模型，支持图像/视频输入；适合召回测试",
    },
    {
        "id": "qwen3.5-plus",
        "label": "MuleRouter Qwen3.5 Plus",
        "note": "MuleRouter 官方 Qwen ID；适合精检/人工窗口回归",
    },
    {
        "id": "qwen3.5-omni-plus",
        "label": "MuleRouter Qwen3.5 Omni Plus",
        "note": "更高质量多模态模型，支持图像/视频输入；成本较高",
    },
    {
        "id": "qwen3.6-plus",
        "label": "MuleRouter Qwen3.6 Plus",
        "note": "MuleRouter 官方 Qwen ID；更强但更贵",
    },
    {
        "id": "qwen/qwen3.6-35b-a3b",
        "label": "Qwen3.6 35B A3B",
        "note": "Qwen 开放权重视觉/视频模型，价格低但准确率需评估",
    },
    {
        "id": "qwen/qwen3.5-flash-02-23",
        "label": "Qwen3.5 Flash",
        "note": "当前实测 OpenRouter/Alibaba 上游 429，暂不建议",
    },
]

INSPECTOR_RECOMMENDED_COARSE_MODEL = "qwen3.6-flash"
INSPECTOR_RECOMMENDED_FINE_MODEL = "qwen3.5-omni-flash"
INSPECTOR_RECOMMENDED_VERIFY_MODEL = "qwen3.6-flash"

def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_progress_line(progress, message):
    stamp = time.strftime("%H:%M:%S", time.localtime())
    return f"[{stamp}] {int(progress):3d}% {message}"

def parse_srt_timestamp(value):
    hours, minutes, remainder = value.split(":", 2)
    seconds, millis = remainder.split(",", 1)
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + (int(millis) / 1000.0)
    )

def get_local_copy_path(container_path: str) -> Path:
    local_dir = get_local_video_dir()
    cp = PurePosixPath(container_path.lstrip("/"))
    return local_dir / str(cp)

def is_false_positive_reason(reason: str) -> bool:
    if not reason:
        return False
    reason_lower = reason.lower()
    has_stocking_mention = any(x in reason_lower for x in ["丝袜", "裤袜", "连裤袜", "袜", "pantyhose", "stocking", "hosiery"])
    negations = [
        "没有穿丝袜", "没穿丝袜", "未穿丝袜", "不穿丝袜", "未穿着丝袜", "没穿着丝袜",
        "没有穿袜子", "没穿袜子", "未穿袜子", "都没有穿", "没有穿戴",
        "没有看到丝袜", "未看到丝袜", "没看到丝袜", "看不到丝袜", "未发现丝袜", "没发现丝袜",
        "不是丝袜", "非丝袜",
        "运动袜", "短袜", "棉袜", "白袜", "白色袜子", "白色的袜子", "白色的袜",
        "可能性不大", "可能没穿", "可能没有穿",
    ]
    for neg in negations:
        if neg in reason_lower:
            return True

    bare_surface_terms = ["裸腿", "裸露"]
    if not has_stocking_mention and any(term in reason_lower for term in bare_surface_terms):
        return True

    # For barefoot negations, only negate if a stocking/sock is NOT explicitly mentioned
    barefoot_negations = ["赤脚", "光脚", "光着脚"]
    if not has_stocking_mention:
        for b_neg in barefoot_negations:
            if b_neg in reason_lower:
                return True
    
    unable = ["无法", "难以", "不确定", "不能", "看不出"]
    judge = ["判断", "确定", "看出", "知道", "清晰分辨", "辨别", "分清", "辨认"]
    has_positive_candidate_language = any(
        term in reason_lower
        for term in ["符合", "候选", "可能性", "可能", "特征", "光滑", "均匀", "丝袜的"]
    )
    for u in unable:
        if u in reason_lower:
            if has_stocking_mention and has_positive_candidate_language:
                continue
            if u == "看不出":
                return True
            for j in judge:
                if j in reason_lower:
                    return True
    return False

STRICT_STOCKING_EVIDENCE_RULES = (
    "严格证据要求（精确模式）：\n"
    "- 不要只因为皮肤光滑、反光、肤色均匀、低光、压缩噪声、距离远，就判断为丝袜/裤袜；这些经常是误报。\n"
    "- 不要根据职业装、裙子、高跟鞋、剧情类型或“通常会穿”来推断丝袜；必须看见视觉证据。\n"
    "- 裙装/高跟鞋下的裸腿不是命中，除非能看到具体丝袜证据。\n"
    "- 黑色长裤、西裤、打底裤、厚裤袜、深色不透明布料都不是目标丝袜；如果裤脚下露出的脚/踝看起来是裸露皮肤，不要返回。\n"
    "- 黑色长裤下方露出光脚通常不是命中，即使脚看起来光滑或画面模糊。\n"
    "- 具体正证据至少包括一种：脚尖缝线/加固脚尖、脚趾或脚背被半透明织物覆盖、袜口/袜边/腰部边缘、织物张力/褶皱/网纹/尼龙纹理，或脚和腿上连续的半透明丝袜层。\n"
    "- 如果脚被鞋完全遮住，腿上也没有袜口、织物纹理或半透明层，就按裸腿处理。\n"
    "- 长裤/西裤场景只有在裤脚下的脚或脚踝能看出被薄丝袜覆盖，并有具体织物证据时才返回。\n"
    "- 如果完全不确定，精确模式下不要返回。\n\n"
)

BAREFOOT_STOCKING_RECALL_RULES = (
    "光脚/脱鞋/看似赤脚的肉色丝袜召回规则：\n"
    "- 这个项目刻意追求高召回，尤其重视“脱鞋后看起来像光脚、但可能穿肉色丝袜/裤袜”的镜头。\n"
    "- 女性坐着、站着、趴着、跪着、上楼、进屋、榻榻米/室内地面/玄关/更衣场景，只要鞋已脱掉或鞋在附近，且脚/腿看起来被一层均匀的肉色或灰色薄层覆盖，就要返回候选。\n"
    "- 即使看不到脚尖缝线，只要脚趾/趾甲细节被柔化或遮住、脚趾缝不清楚、脚底/脚背呈连续光滑的肉色/灰色覆盖层，也要返回候选。\n"
    "- 特别关注脚底、脚背、脚尖、脚踝、袜口、裤脚下露出的脚、趴在地上或弯腰时露出的脚底。这些经常是最关键画面。\n"
    "- 中景、远景、模糊镜头也不要轻易丢弃；弱但有用的候选用 confidence='low'，较可能但缺少缝线/袜口证据用 confidence='medium'。用户会人工复核。\n"
    "- 只有在能清楚看到裸露脚趾/趾甲/自然皮肤纹理、普通棉袜/运动袜，或目标脚腿区域完全不可见时，才排除。\n\n"
)

# Simplified coarse prompt (default for the coarse screenshot pass). Keeps the
# high-recall intent for genuinely ambiguous "脱鞋后像光脚但可能肉色丝袜" cases,
# but names the obvious non-targets (sandals, trousers, clearly-bare feet,
# outdoor/far) so the model stops flooding on them — far cleaner than the old
# ~600-char nested-rule prompt, and much better at rejecting obvious junk.
SIMPLE_COARSE_PROMPT = (
    "下面是一组视频帧，每帧带 Frame ID。逐帧找出画面里可能有女性穿着薄款、半透明的肉色/肤色或灰色丝袜/裤袜、露出腿或脚的镜头。"
    "包括脱鞋后看起来像光脚、但脚/腿可能是肉色丝袜的情况（这种最容易漏，宁可返回交给后续判断）。\n"
    "黑丝/白丝/彩色丝袜、不透明打底裤/leggings、棉袜/运动袜不是目标；腿完全被长裤遮住、或只是远景里有人而看不清腿脚的，也不用返回。\n"
    "reason 用中文如实描述你看到的（穿什么、露不露腿脚、是否凉鞋/长裤/裸足、皮肤质感与光泽等），由后续步骤判断有没有丝袜，你不要下最终结论也不要打分。\n"
    "只返回严格 JSON：{\"detections\":[{\"frame_id\":\"frame_0001\",\"reason\":\"...\"}]}，无候选返回 {\"detections\":[]}。"
)

CONCRETE_HOSIERY_EVIDENCE_TERMS = (
    "袜口", "袜边", "袜带", "脚尖缝", "脚趾缝", "缝头", "缝线", "接缝", "加固",
    "织物", "布料", "纤维", "网纹", "网眼", "纹理", "褶皱", "皱褶", "罗纹",
    "覆盖", "包裹", "包覆", "半透明层", "透明织物", "尼龙纹理", "尼龙层",
    "脚趾被", "脚背被", "脚部被", "toe seam", "reinforced toe", "waistband",
    "ankle band", "fabric edge", "mesh", "nylon texture", "sheer fabric",
    "covered toes", "covered foot", "fabric wrinkles",
)

WEAK_HOSIERY_EVIDENCE_TERMS = (
    "光滑", "均匀", "光泽", "反光", "肤色", "职业", "商务", "通常", "习惯",
    "符合", "疑似", "可能", "看起来", "推断", "低光", "昏暗", "模糊",
    "smooth", "shiny", "uniform", "likely", "probably", "business attire",
)

def has_concrete_hosiery_evidence(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text or is_false_positive_reason(text):
        return False
    return any(term.lower() in text for term in CONCRETE_HOSIERY_EVIDENCE_TERMS)

def is_weak_hosiery_claim(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    mentions_hosiery = any(x in text for x in ["丝袜", "裤袜", "连裤袜", "pantyhose", "stocking", "hosiery"])
    has_weak = any(term.lower() in text for term in WEAK_HOSIERY_EVIDENCE_TERMS)
    return mentions_hosiery and has_weak and not has_concrete_hosiery_evidence(text)

# LEGACY: precision-era predicate, no longer called by any pipeline path
# (superseded by keep_coarse_detection + the semantic reason filter). Retained
# only because an external regression test exercises it directly.
def should_keep_precision_detection(detection: dict) -> bool:
    reason = str((detection or {}).get("reason") or "")
    if is_false_positive_reason(reason):
        return False
    return has_concrete_hosiery_evidence(reason)

def should_keep_model_detection(detection: dict, require_concrete_evidence: bool) -> bool:
    reason = str((detection or {}).get("reason") or "")
    if is_false_positive_reason(reason):
        return False
    if require_concrete_evidence:
        return has_concrete_hosiery_evidence(reason)
    return True

def should_verify_detection(detection: dict, policy: str) -> bool:
    policy = str(policy or "medium").strip().lower()
    if policy in {"0", "false", "off", "none"}:
        return False
    if policy in {"1", "true", "yes", "on", "all"}:
        return True
    conf = str((detection or {}).get("confidence") or "medium").strip().lower()
    if policy in {"medium", "medium_low", "uncertain"}:
        reason = (detection or {}).get("reason") or ""
        return conf != "high" or not has_concrete_hosiery_evidence(reason) or is_weak_hosiery_claim(reason)
    return True

def dedupe_temporal_detections(detections, window_seconds=3.0):
    if not detections:
        return []
    try:
        window = max(0.0, float(window_seconds))
    except (TypeError, ValueError):
        window = 3.0
    if window <= 0:
        return detections

    conf_order = {"high": 1, "medium": 2, "low": 3}
    by_time = sorted(detections, key=lambda x: float(x.get("seconds") or 0.0))
    groups = []
    current = []
    last_sec = None
    for det in by_time:
        sec = float(det.get("seconds") or 0.0)
        if last_sec is None or sec - last_sec <= window:
            current.append(det)
        else:
            groups.append(current)
            current = [det]
        last_sec = sec
    if current:
        groups.append(current)

    winners = []
    for group in groups:
        winners.append(min(
            group,
            key=lambda x: (
                conf_order.get(str(x.get("confidence") or "medium").lower(), 99),
                0 if has_concrete_hosiery_evidence(x.get("reason") or "") else 1,
                float(x.get("seconds") or 0.0),
            ),
        ))
    winners.sort(key=lambda x: (conf_order.get(str(x.get("confidence") or "medium").lower(), 99), x["seconds"]))
    return winners

# --------------------------------------------------------------------------
# Pure post-processing helpers.
#
# These mirror the per-stage keep/skip logic and the final sort+dedupe that
# run_inspection applies to raw model detections. They are deliberately pure
# (no I/O, no settings reads) so the same logic drives both live inspection
# and the offline replay harness (inspector_replay.py) — replaying a saved
# trace.json through these reproduces a run's final detections deterministically.
# --------------------------------------------------------------------------

MALE_SUBJECT_TERMS = ("男人", "男子", "男性", "男士", "男孩", "男生", "大叔", "老头", "a man", "the man", " male ")
FEMALE_SUBJECT_TERMS = ("女", "她", "woman", "women", "girl", "lady", "female", "her ")

def is_excluded_subject_reason(reason: str) -> bool:
    """True when the reason clearly describes a male-only subject.

    The target is female legs/feet; the model occasionally flags a man wearing
    nude hosiery (a real but unwanted hit). We only exclude when a male subject
    is named AND no female subject is mentioned, so mixed-gender scenes survive.
    """
    text = str(reason or "").strip().lower()
    if not text:
        return False
    has_male = any(t.strip().lower() in text for t in MALE_SUBJECT_TERMS)
    if not has_male:
        return False
    has_female = any(t.strip().lower() in text for t in FEMALE_SUBJECT_TERMS)
    return not has_female

_WEAK_COARSE_PATTERNS = (
    re.compile(r"高召回|召回规则|召回原则|召回策略"),   # model admits it's only included per the recall rule
    re.compile(r"凉鞋|拖鞋"),                            # sandals/slippers -> not the target
    re.compile(r"长裤|西裤|短裤|牛仔裤"),                # trousers, legs covered
    re.compile(r"脚趾.{0,4}(清晰|可见|外露|明显)|趾甲.{0,3}(清晰|可见)|脚趾甲"),  # clear toe detail = bare (hosiery softens toes)
)

def is_weak_coarse_candidate(reason: str) -> bool:
    """True for a coarse candidate the model only kept out of recall bias.

    These are the dominant false-positive class on bare-foot/sandal/trouser-heavy
    videos: the recall-tuned coarse prompt makes the model write "per the
    high-recall rule …" or describe visible toes / sandals / trousers, yet still
    return the frame. We drop them only when there is NO concrete hosiery
    evidence, so genuine close-ups (toe seam, waistband, mesh) always survive.
    """
    text = str(reason or "").strip()
    if not text or has_concrete_hosiery_evidence(text):
        return False
    return any(p.search(text) for p in _WEAK_COARSE_PATTERNS)

def keep_coarse_detection(det: dict, *, extreme_recall: bool, hybrid: bool, is_two_pass: bool, exclude_male_subject: bool = True, drop_weak_coarse: bool = False) -> bool:
    """Per-stage keep predicate for a COARSE-pass detection.

    - hybrid coarse: only the false-positive language filter applies (coarse
      candidates merely seed fine windows).
    - screenshot coarse: false-positive filter, plus a low-confidence gate when
      this is a single-pass run (no fine pass to re-confirm).
    """
    # Wrong-subject (male-only) exclusion applies even under extreme recall: a
    # man's legs are never the target, so dropping them costs no target recall.
    if exclude_male_subject and is_excluded_subject_reason(det.get("reason", "")):
        return False
    if drop_weak_coarse and is_weak_coarse_candidate(det.get("reason", "")):
        return False
    if extreme_recall:
        return True
    if is_false_positive_reason(det.get("reason", "")):
        return False
    if not hybrid and not is_two_pass:
        conf = str(det.get("confidence", "low")).strip().lower()
        if conf == "low":
            return False
    return True

def keep_fine_detection(det: dict, *, extreme_recall: bool, require_concrete_evidence: bool, keep_low_fine: bool, exclude_male_subject: bool = True) -> bool:
    """Per-stage keep predicate for a FINE-pass detection (screenshot or video).

    Shared by the screenshot fine loop and the hybrid video fine pass — both
    apply the false-positive + evidence filter (``should_keep_model_detection``)
    then drop low-confidence hits unless ``keep_low_fine`` is set.
    """
    if exclude_male_subject and is_excluded_subject_reason(det.get("reason", "")):
        return False
    if extreme_recall:
        return True
    if not should_keep_model_detection(det, require_concrete_evidence):
        return False
    conf = str(det.get("confidence", "low")).strip().lower()
    if conf == "low" and not keep_low_fine:
        return False
    return True

def fine_video_detection_seconds(time_str, start_seconds: float, duration: float, rep_offset: float) -> tuple[float, float]:
    """Map a fine-video model ``time``/``time-range`` to absolute seconds.

    Returns ``(abs_sec, source_abs_sec)`` where ``abs_sec`` is the representative
    timestamp to extract a still from (midpoint of a range, else start+offset),
    and ``source_abs_sec`` is the segment-relative start mapped to absolute time.
    """
    rel_start_sec, rel_end_sec = timestamp_range_to_seconds(time_str)
    if rel_end_sec is not None and rel_end_sec > rel_start_sec:
        rel_sec = (rel_start_sec + rel_end_sec) / 2.0
    else:
        rel_sec = min(rel_start_sec + rep_offset, max(0.0, duration - 0.2))
    return start_seconds + rel_sec, start_seconds + rel_start_sec

def annotate_scene_support(detections: list[dict], support_pool: list[dict], window_seconds: float) -> list[dict]:
    """Annotate each detection with ``scene_support`` = how many items in
    ``support_pool`` fall within ``window_seconds`` of it.

    The support pool is normally the dense coarse-candidate set: a genuine
    stockings scene lights up many consecutive coarse frames, while a transient
    single-frame false positive is supported by only itself. Mutates + returns
    ``detections``.
    """
    try:
        window = max(0.0, float(window_seconds))
    except (TypeError, ValueError):
        window = 0.0
    pool_secs = sorted(float(d.get("seconds") or 0.0) for d in (support_pool or []))
    import bisect
    for det in detections:
        s = float(det.get("seconds") or 0.0)
        lo = bisect.bisect_left(pool_secs, s - window)
        hi = bisect.bisect_right(pool_secs, s + window)
        det["scene_support"] = hi - lo
    return detections

def filter_by_scene_support(detections: list[dict], *, min_support: int, protect_high: bool = True) -> list[dict]:
    """Drop detections with ``scene_support`` < ``min_support``.

    With ``protect_high`` (default), high-confidence detections survive even
    when isolated, so a single clear close-up is never discarded. Requires
    ``annotate_scene_support`` to have run first; if ``min_support`` <= 1 this is
    a no-op.
    """
    if min_support is None or min_support <= 1:
        return detections
    kept = []
    for det in detections:
        if det.get("scene_support", 1) >= min_support:
            kept.append(det)
        elif protect_high and str(det.get("confidence") or "").strip().lower() == "high":
            kept.append(det)
    return kept

def finalize_detections(detections: list[dict], *, dedup_window) -> list[dict]:
    """Sort by (confidence, seconds) then collapse near-duplicate scene frames.

    This is the tail of run_inspection: a stable confidence-then-time sort
    followed by temporal dedupe within ``dedup_window`` seconds.
    """
    conf_order = {"high": 1, "medium": 2, "low": 3}
    ordered = sorted(
        detections,
        key=lambda x: (conf_order.get(x.get("confidence", "medium"), 99), x["seconds"]),
    )
    return dedupe_temporal_detections(ordered, dedup_window)

def image_dhash(image_path, hash_size=8) -> int | None:
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), LANCZOS)
            pixels = img.tobytes()
    except Exception:
        return None

    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits <<= 1
            if pixels[offset + col] > pixels[offset + col + 1]:
                bits |= 1
    return bits

def hamming_distance(a: int | None, b: int | None) -> int:
    if a is None or b is None:
        return 64
    return int(a ^ b).bit_count()

def dedupe_similar_frames(frames, threshold=6):
    if not frames:
        return []
    kept = []
    seen_hashes = []
    for frame in frames:
        dhash = image_dhash(frame.get("file_path"))
        frame["local_dhash"] = dhash
        if dhash is None:
            kept.append(frame)
            continue
        if any(hamming_distance(dhash, prev) <= threshold for prev in seen_hashes):
            continue
        seen_hashes.append(dhash)
        kept.append(frame)
    return kept

def safe_trace_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_") or "trace"

def _corpus_protected_container_paths() -> set:
    """Container paths annotated in the regression corpus, whose job traces must
    survive job cleanup so the offline replay harness can still reconstruct them."""
    try:
        from . import corpus as inspector_corpus  # lazy: avoid import cost on hot paths
        return {
            bucket.get("container_path")
            for bucket in inspector_corpus.corpus_by_video().values()
            if bucket.get("container_path")
        }
    except Exception:
        return set()

def manual_marks_jsonl_path() -> Path:
    return get_local_video_dir() / "inspections" / "manual_marks.jsonl"

def list_manual_marks(container_path: str | None = None) -> list[dict]:
    path = manual_marks_jsonl_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if container_path and item.get("container_path") != container_path:
            continue
        out.append(item)
    out.sort(key=lambda x: (float(x.get("seconds") or 0.0), x.get("created_at") or 0))
    return out

def append_manual_mark(mark: dict) -> dict:
    path = manual_marks_jsonl_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    mark = dict(mark)
    mark.setdefault("mark_id", str(uuid.uuid4()))
    mark.setdefault("created_at", int(time.time()))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(mark, ensure_ascii=False) + "\n")
    return mark

def cluster_manual_marks(marks: list[dict], merge_window_seconds: float = 8.0) -> list[dict]:
    positives = [
        dict(m) for m in marks
        if (m.get("label") or "positive") == "positive" and m.get("container_path")
    ]
    positives.sort(key=lambda m: (m.get("container_path") or "", float(m.get("seconds") or 0.0)))
    clusters: list[dict] = []
    for mark in positives:
        seconds = float(mark.get("seconds") or 0.0)
        if (
            clusters
            and clusters[-1]["container_path"] == mark.get("container_path")
            and seconds - clusters[-1]["end_seconds"] <= merge_window_seconds
        ):
            cluster = clusters[-1]
            cluster["marks"].append(mark)
            cluster["mark_ids"].append(mark.get("mark_id"))
            cluster["start_seconds"] = min(cluster["start_seconds"], seconds)
            cluster["end_seconds"] = max(cluster["end_seconds"], seconds)
            cluster["seconds"] = round((cluster["start_seconds"] + cluster["end_seconds"]) / 2.0, 3)
            if mark.get("screenshot_file"):
                cluster["screenshot_file"] = mark.get("screenshot_file")
            continue
        clusters.append({
            "case_id": mark.get("mark_id") or str(uuid.uuid4()),
            "container_path": mark.get("container_path"),
            "seconds": seconds,
            "start_seconds": seconds,
            "end_seconds": seconds,
            "timestamp": mark.get("timestamp"),
            "screenshot_file": mark.get("screenshot_file"),
            "note": mark.get("note") or "",
            "mark_ids": [mark.get("mark_id")],
            "marks": [mark],
        })
    return clusters

def cluster_negative_marks(marks: list[dict], merge_window_seconds: float = 8.0) -> list[dict]:
    """Cluster human-confirmed NEGATIVE marks into negative events.

    Mirrors ``cluster_manual_marks`` but for ``label == "negative"`` rows. These
    define regions where an AI detection is a genuine false positive.
    """
    negatives = [
        dict(m) for m in marks
        if (m.get("label") or "").strip().lower() == "negative" and m.get("container_path")
    ]
    negatives.sort(key=lambda m: (m.get("container_path") or "", float(m.get("seconds") or 0.0)))
    clusters: list[dict] = []
    for mark in negatives:
        seconds = float(mark.get("seconds") or 0.0)
        if (
            clusters
            and clusters[-1]["container_path"] == mark.get("container_path")
            and seconds - clusters[-1]["end_seconds"] <= merge_window_seconds
        ):
            cluster = clusters[-1]
            cluster["start_seconds"] = min(cluster["start_seconds"], seconds)
            cluster["end_seconds"] = max(cluster["end_seconds"], seconds)
            cluster["seconds"] = round((cluster["start_seconds"] + cluster["end_seconds"]) / 2.0, 3)
            cluster["mark_ids"].append(mark.get("mark_id"))
            continue
        clusters.append({
            "case_id": mark.get("mark_id") or str(uuid.uuid4()),
            "container_path": mark.get("container_path"),
            "seconds": seconds,
            "start_seconds": seconds,
            "end_seconds": seconds,
            "mark_ids": [mark.get("mark_id")],
        })
    return clusters


def match_detections_to_events(
    positive_events: list[dict],
    negative_events: list[dict],
    detections: list[dict],
    *,
    tolerance_seconds: float = 30.0,
) -> dict:
    """Greedy one-to-one matching of detections against ground-truth events.

    Each positive event claims at most one (nearest, unclaimed) detection within
    ``tolerance_seconds`` of its [start, end] span — so a single detection can no
    longer satisfy several events. Leftover detections are classified:
    ``false_positive`` if they fall near a negative event, else ``unknown``
    (an AI hit in a region the human never labeled — not counted against precision).

    Greedy nearest-first, not optimal bipartite; deterministic and adequate at
    this scale.
    """
    dets = []
    for i, det in enumerate(detections):
        if det.get("seconds") is None:
            continue
        d = dict(det)
        d["_idx"] = i
        d["seconds"] = float(d.get("seconds") or 0.0)
        dets.append(d)

    claimed: set[int] = set()
    evaluated_cases: list[dict] = []
    true_positives = 0
    for case in sorted(positive_events, key=lambda c: float(c.get("seconds") or 0.0)):
        lo = float(case["start_seconds"]) - tolerance_seconds
        hi = float(case["end_seconds"]) + tolerance_seconds
        center = float(case["seconds"])
        candidates = sorted(
            (d for d in dets if d["_idx"] not in claimed and lo <= d["seconds"] <= hi),
            key=lambda d: abs(d["seconds"] - center),
        )
        best = candidates[0] if candidates else None
        if best is not None:
            claimed.add(best["_idx"])
            true_positives += 1
        evaluated_cases.append({
            **{k: v for k, v in case.items() if k != "marks"},
            "matched": best is not None,
            "matched_detection": {k: v for k, v in best.items() if k != "_idx"} if best else None,
            "delta_seconds": abs(best["seconds"] - center) if best else None,
        })

    leftover = [d for d in dets if d["_idx"] not in claimed]
    false_positives, unknown = [], []
    for d in leftover:
        near_negative = any(
            float(n["start_seconds"]) - tolerance_seconds <= d["seconds"] <= float(n["end_seconds"]) + tolerance_seconds
            for n in negative_events
        )
        (false_positives if near_negative else unknown).append({k: v for k, v in d.items() if k != "_idx"})

    positives = len(positive_events)
    false_negatives = positives - true_positives
    recall = true_positives / positives if positives else None
    # Honest precision needs labeled negatives; without them it is undefined.
    precision_denom = true_positives + len(false_positives)
    precision = (true_positives / precision_denom) if (negative_events and precision_denom) else None
    # Legacy optimistic estimate: treat every unmatched detection as a miss.
    est_denom = true_positives + len(leftover)
    precision_estimate = (true_positives / est_denom) if est_denom else None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "positive_events": positives,
        "negative_events": len(negative_events),
        "detections": len(dets),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": len(false_positives),
        "unknown_detections": len(unknown),
        "unmatched_detections": len(leftover),
        "recall": recall,
        "precision": precision,
        "precision_estimate": precision_estimate,
        "f1": f1,
        "cases": evaluated_cases,
        "false_positive_items": false_positives,
        "unknown_detection_items": unknown,
    }


def evaluate_inspector_result(
    result: dict,
    marks: list[dict],
    *,
    tolerance_seconds: float = 30.0,
    merge_window_seconds: float = 8.0,
) -> dict:
    container_path = result.get("container_path")
    scoped_marks = [
        m for m in marks
        if not container_path or m.get("container_path") == container_path
    ]
    cases = cluster_manual_marks(scoped_marks, merge_window_seconds=merge_window_seconds)
    negative_events = cluster_negative_marks(scoped_marks, merge_window_seconds=merge_window_seconds)
    detections = result.get("visual_detections") or []
    metrics = match_detections_to_events(
        cases, negative_events, detections, tolerance_seconds=tolerance_seconds,
    )
    return {
        "container_path": container_path,
        "video_file": result.get("video_file"),
        "tolerance_seconds": tolerance_seconds,
        "merge_window_seconds": merge_window_seconds,
        "manual_marks": len(scoped_marks),
        # backward-compatible key kept for existing callers / UI
        "unmatched_detection_items": metrics["false_positive_items"] + metrics["unknown_detection_items"],
        **metrics,
    }


def evaluate_against_corpus(
    detections_by_video: dict[str, list[dict]],
    corpus_by_video: dict[str, dict],
    *,
    tolerance_seconds: float = 30.0,
    merge_window_seconds: float = 8.0,
) -> dict:
    """Aggregate scoring across every annotated video in the corpus.

    ``detections_by_video`` maps a corpus ``video_id`` to that video's final
    detections (from a live run or a replay). ``corpus_by_video`` is
    ``inspector_corpus.corpus_by_video()``. Returns per-video metrics plus a
    pooled aggregate (micro-averaged recall/precision/F1).
    """
    def _to_events(rows: list[dict]) -> list[dict]:
        pts = sorted(rows, key=lambda r: float(r.get("seconds") or 0.0))
        events: list[dict] = []
        for r in pts:
            sec = float(r.get("seconds") or 0.0)
            if events and sec - events[-1]["end_seconds"] <= merge_window_seconds:
                ev = events[-1]
                ev["end_seconds"] = max(ev["end_seconds"], sec)
                ev["seconds"] = round((ev["start_seconds"] + ev["end_seconds"]) / 2.0, 3)
                ev["ids"].append(r.get("id"))
            else:
                events.append({"seconds": sec, "start_seconds": sec, "end_seconds": sec, "ids": [r.get("id")]})
        return events

    per_video: dict[str, dict] = {}
    totals = {"true_positives": 0, "false_negatives": 0, "false_positives": 0,
              "unknown_detections": 0, "positive_events": 0, "negative_events": 0, "detections": 0}
    for vid, bucket in corpus_by_video.items():
        pos_events = _to_events(bucket.get("positives") or [])
        neg_events = _to_events(bucket.get("negatives") or [])
        dets = detections_by_video.get(vid, [])
        m = match_detections_to_events(pos_events, neg_events, dets, tolerance_seconds=tolerance_seconds)
        per_video[vid] = m
        for k in totals:
            totals[k] += m.get(k, 0) or 0

    tp, fp = totals["true_positives"], totals["false_positives"]
    pos = totals["positive_events"]
    recall = tp / pos if pos else None
    precision = tp / (tp + fp) if (totals["negative_events"] and (tp + fp)) else None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "tolerance_seconds": tolerance_seconds,
        "merge_window_seconds": merge_window_seconds,
        "videos": len(corpus_by_video),
        "videos_with_detections": sum(1 for v in detections_by_video if detections_by_video[v]),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        **totals,
        "per_video": per_video,
    }


def compare_eval_runs(baseline: dict, candidate: dict) -> dict:
    """Diff two corpus evaluations (each from ``evaluate_against_corpus``).

    Surfaces headline deltas plus, per video, which positive events newly pass
    (fixed) and which regressed (was matched in baseline, now missed).
    """
    def _matched_event_keys(metrics: dict) -> set[tuple]:
        keys = set()
        for vid, m in (metrics.get("per_video") or {}).items():
            for case in m.get("cases") or []:
                if case.get("matched"):
                    keys.add((vid, round(float(case.get("seconds") or 0.0), 1)))
        return keys

    base_hits = _matched_event_keys(baseline)
    cand_hits = _matched_event_keys(candidate)
    fixed = sorted(cand_hits - base_hits)
    regressed = sorted(base_hits - cand_hits)

    def _delta(key):
        b, c = baseline.get(key), candidate.get(key)
        if b is None or c is None:
            return None
        return round(c - b, 4)

    return {
        "recall_delta": _delta("recall"),
        "precision_delta": _delta("precision"),
        "f1_delta": _delta("f1"),
        "true_positives_delta": (candidate.get("true_positives", 0) - baseline.get("true_positives", 0)),
        "false_positives_delta": (candidate.get("false_positives", 0) - baseline.get("false_positives", 0)),
        "fixed_events": [{"video_id": v, "seconds": s} for v, s in fixed],
        "regressed_events": [{"video_id": v, "seconds": s} for v, s in regressed],
        "is_regression": len(regressed) > 0
        or (_delta("recall") is not None and _delta("recall") < 0)
        or (_delta("precision") is not None and _delta("precision") < 0),
    }

def timestamp_to_seconds(ts_str):
    ts = str(ts_str or "").strip()
    if "-" in ts:
        ts = ts.split("-", 1)[0].strip()
    try:
        parts = list(map(int, ts.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except Exception:
        pass
    return 0.0

def timestamp_range_to_seconds(ts_str):
    text = str(ts_str or "").strip()
    if not text:
        return 0.0, None
    parts = re.split(r"\s*(?:-|–|—|~|〜|至|到)\s*", text, maxsplit=1)
    start = timestamp_to_seconds(parts[0])
    if len(parts) > 1 and parts[1].strip():
        return start, timestamp_to_seconds(parts[1])
    return start, None

def unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        key = round(float(value), 3)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return out

def is_truthy_setting(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

def get_int_setting(name: str, default: int, min_value: int = 1, max_value: int | None = None) -> int:
    try:
        value = int(get_setting(name, default))
    except (TypeError, ValueError):
        value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def get_float_setting(name: str, default: float, min_value: float = 0.0, max_value: float | None = None) -> float:
    try:
        value = float(get_setting(name, default))
    except (TypeError, ValueError):
        value = default
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value

def api_provider_label_from_url(url: str) -> str:
    text = str(url or "").lower()
    if "mulerouter" in text:
        return "MuleRouter"
    if "openrouter" in text:
        return "OpenRouter"
    return "Vision API"

def format_openrouter_error(exc: requests.exceptions.HTTPError) -> str:
    return format_api_error(exc, "OpenRouter")

def format_api_error(exc: requests.exceptions.HTTPError, provider_label: str = "Vision API") -> str:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = ""
    code = ""
    metadata = None
    raw_payload = None
    retry_after = ""
    if response is not None:
        try:
            retry_after = response.headers.get("Retry-After", "") if response.headers else ""
        except Exception:
            retry_after = ""
        try:
            payload = response.json()
            raw_payload = payload
            err = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err, dict):
                code = str(err.get("code") or "")
                message = str(err.get("message") or err.get("code") or "")
                metadata = err.get("metadata")
            elif err:
                message = str(err)
        except Exception:
            try:
                message = (response.text or "").strip()
                raw_payload = message
            except Exception:
                message = ""
    if not message:
        message = str(exc)
    details = []
    if code:
        details.append(f"code={code}")
    if retry_after:
        details.append(f"retry_after={retry_after}s")
    if metadata:
        try:
            details.append("metadata=" + json.dumps(metadata, ensure_ascii=False, default=str))
        except Exception:
            details.append(f"metadata={metadata}")
    if raw_payload is not None and not metadata:
        try:
            details.append("raw=" + json.dumps(raw_payload, ensure_ascii=False, default=str))
        except Exception:
            details.append(f"raw={raw_payload}")
    suffix = f" ({'; '.join(details)})" if details else ""
    if status:
        return f"{provider_label} HTTP {status}: {message}{suffix}"
    return f"{provider_label} request failed: {message}{suffix}"

def explain_inspector_error(message: str) -> dict:
    text = str(message or "")
    lower = text.lower()
    info = {
        "summary": "检测失败",
        "why": text or "任务运行时出现未知错误。",
        "suggestions": [],
        "recommended_model": INSPECTOR_RECOMMENDED_COARSE_MODEL,
        "recommended_fine_model": INSPECTOR_RECOMMENDED_FINE_MODEL,
    }
    if "quota" in lower or "token-limit" in lower or "exceeded your current quota" in lower:
        info["summary"] = "模型供应商额度/速率限制"
        info["why"] = (
            "请求已经发到当前视觉供应商，但底层模型供应商返回了额度或 token 限制。"
            "MuleRouter 上的 Qwen 通常由阿里 Model Studio 承载，所以错误里会出现 alibabacloud 链接。"
        )
        info["suggestions"] = [
            "暂停一段时间后再重试，等待上游额度窗口恢复。",
            "降低采样密度或 fine video 段数。",
            "切换到 MuleRouter/OpenRouter 上的其他视觉模型分摊额度。",
        ]
    elif "openrouter http 429" in lower or "mulerouter http 429" in lower or "provider returned error" in lower:
        info["summary"] = "视觉模型供应商限流"
        info["why"] = (
            "当前视觉 API 已接收请求，但转发给底层模型供应商时收到限流。"
            "这通常是模型供应商临时繁忙、账号/模型限额触发，或当前模型池拥堵。"
        )
        info["suggestions"] = [
            "等几十秒后重试。",
            "切换到 Qwen / OpenAI / Claude 等不同供应商的视觉模型。",
            "避免同时启动多个检测任务；必要时把采样间隔调大一些。",
        ]
    elif "openrouter http 403" in lower or "mulerouter http 403" in lower or "key limit exceeded" in lower:
        info["summary"] = "视觉 API key 额度或权限不足"
        info["why"] = "当前视觉 API 拒绝了请求，常见原因是 key 额度耗尽、模型无权限或账号限制。"
        info["suggestions"] = [
            "检查当前 provider 的 key 余额、额度和模型权限。",
            "换一个允许视觉输入的模型。",
        ]
    elif "openrouter http" in lower or "mulerouter http" in lower:
        info["summary"] = "视觉 API 请求失败"
        info["why"] = "当前视觉 API 返回了 HTTP 错误，错误详情见原始信息。"
        info["suggestions"] = [
            "稍后重试。",
            "如果同一模型持续失败，切换到另一个视觉模型。",
        ]
    elif "no frames extracted" in lower:
        info["summary"] = "没有成功提取视频帧"
        info["why"] = "ffmpeg/视频解码没有从该文件中提取到可分析截图。"
        info["suggestions"] = [
            "确认视频文件可播放。",
            "换一个采样间隔后重试。",
        ]
    else:
        info["suggestions"] = [
            "查看原始错误。",
            "如果错误来自模型供应商，尝试切换模型后重试。",
        ]
    return info

def post_openrouter_with_retry(url, headers, payload, timeout, attempts=3):
    last_exc = None
    provider_label = api_provider_label_from_url(url)
    retryable = (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.ReadTimeout,
        requests.exceptions.Timeout,
    )
    # Transient HTTP statuses worth retrying. 429/5xx are the usual suspects;
    # 400 is included because MuleRouter intermittently emits a spurious
    # "Bad request" under burst concurrency that succeeds on a plain retry
    # (the identical payload reproduces clean), so treat it as transient here.
    retryable_status = {400, 429, 500, 502, 503, 504}
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in retryable_status and attempt < attempts:
                    last_exc = exc
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(format_api_error(exc, provider_label)) from exc
            return response
        except RuntimeError:
            raise
        except retryable as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{provider_label} request failed after {attempts} attempts: {last_exc}") from last_exc

def is_skippable_provider_content_error(exc) -> bool:
    text = str(exc or "").lower()
    return (
        "datainspectionfailed" in text
        or "inappropriate content" in text
        or "input image data may contain inappropriate content" in text
    )

def is_quota_limit_error(exc) -> bool:
    text = str(exc or "").lower()
    return (
        "exceeded your current quota" in text
        or "token-limit" in text
        or "quota" in text
    )

class VideoInspector:
    def __init__(self):
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.openrouter_referer = "https://github.com/ldy-mitbbs/sheerscan"
        self.openrouter_app_title = "sheerscan"
        self._trace_lock = threading.Lock()  # serialize trace.json writes across parallel batches

    def _api_provider(self) -> str:
        provider = str(get_setting("INSPECTOR_API_PROVIDER", "mulerouter") or "mulerouter").strip().lower()
        return provider if provider in {"openrouter", "mulerouter"} else "openrouter"

    def _api_base_url(self) -> str:
        if self._api_provider() == "mulerouter":
            default_base = "https://api.mulerouter.ai/vendors/openai/v1"
            return str(get_setting("MULEROUTER_BASE_URL", default_base) or default_base)
        return str(get_setting("OPENROUTER_BASE_URL", self.openrouter_base_url) or self.openrouter_base_url)

    def _api_key(self) -> str:
        if self._api_provider() == "mulerouter":
            return get_secret("MULEROUTER_API_KEY") or os.environ.get("MULEROUTER_API_KEY") or ""
        return get_secret("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""

    def _normalize_model_name(self, model_name: str) -> str:
        model = str(model_name or "").strip()
        if self._api_provider() != "mulerouter":
            return model
        mappings = {
            "qwen/qwen3.6-flash": "qwen3.6-flash",
            "qwen/qwen3.6-plus": "qwen3.6-plus",
            "qwen/qwen3.5-plus-20260420": "qwen3.5-plus",
            "qwen/qwen3.5-plus-02-15": "qwen3.5-plus",
            "qwen/qwen3.5-flash-02-23": "qwen3.5-flash",
            "qwen/qwen3-vl-flash": "qwen3-vl-flash",
            "qwen/qwen3.5-omni-flash": "qwen3.5-omni-flash",
            "qwen/qwen3.5-omni-plus": "qwen3.5-omni-plus",
        }
        return mappings.get(model, model)

    def _append_trace_event(self, job_dir: Path, event: dict):
        trace_path = Path(job_dir) / "trace.json"
        with self._trace_lock:
            events = []
            if trace_path.exists():
                try:
                    events = json.loads(trace_path.read_text(encoding="utf-8"))
                    if not isinstance(events, list):
                        events = []
                except Exception:
                    events = []
            event = dict(event)
            event.setdefault("created_at", int(time.time()))
            events.append(event)
            trace_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    def _copy_trace_frame(self, frame, job_dir: Path, trace_name: str):
        frames_dir = Path(job_dir) / "debug" / "frames" / safe_trace_name(trace_name)
        frames_dir.mkdir(parents=True, exist_ok=True)
        src = Path(frame["file_path"])
        dest_name = f"{safe_trace_name(frame.get('id'))}_{safe_trace_name(frame.get('timestamp_str'))}.jpg"
        dest = frames_dir / dest_name
        if not dest.exists():
            shutil.copy2(src, dest)
        return {
            "frame_id": frame.get("id"),
            "timestamp": frame.get("timestamp_str"),
            "seconds": frame.get("seconds"),
            "artifact": str(dest.relative_to(job_dir)),
        }

    def _load_local_clip(self, model_name):
        state = _LOCAL_CLIP_STATE
        if state["available"] and state["model_name"] == model_name:
            return state
        if state["available"] is False and state["model_name"] == model_name:
            return state

        state.update({
            "available": False,
            "error": "",
            "model_name": model_name,
            "device": "cpu",
            "model": None,
            "processor": None,
            "torch": None,
        })
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            state.update({
                "available": True,
                "device": device,
                "model": model,
                "processor": processor,
                "torch": torch,
            })
        except Exception as exc:
            state["error"] = str(exc)
        return state

    def _clip_prefilter_frames(self, frames, model_name, min_score, top_k, batch_size, progress_cb=None):
        state = self._load_local_clip(model_name)
        if not state.get("available"):
            raise RuntimeError(
                "Local CLIP prefilter is enabled but unavailable: "
                f"{state.get('error') or 'CLIP dependencies are not installed.'}"
            )

        prompts = [
            "a video frame showing legs",
            "a video frame showing feet",
            "a video frame showing shoes and ankles",
            "a video frame showing stockings or pantyhose",
            "a video frame showing socks",
            "a video frame showing a full body person",
            "a video frame without people, legs, or feet",
            "a close-up of a face or upper body only",
            "a landscape, room, food, object, or text screen",
        ]
        relevant_count = 6
        processor = state["processor"]
        model = state["model"]
        torch = state["torch"]
        device = state["device"]
        scored = []
        total = len(frames)
        batch_size = max(1, int(batch_size or 4))

        for start in range(0, total, batch_size):
            batch = frames[start:start + batch_size]
            images = []
            valid_frames = []
            for frame in batch:
                try:
                    images.append(Image.open(frame["file_path"]).convert("RGB"))
                    valid_frames.append(frame)
                except Exception:
                    frame["local_prefilter_score"] = 1.0
                    frame["local_prefilter_reason"] = "image-open-failed-keep"
                    scored.append(frame)
            if not images:
                continue
            if progress_cb:
                progress_cb(17, f"Local CLIP prefilter: {min(start + len(batch), total)}/{total} frames...")
            try:
                inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1).detach().cpu().tolist()
            finally:
                for image in images:
                    try:
                        image.close()
                    except Exception:
                        pass

            for frame, row in zip(valid_frames, probs):
                relevant = max(row[:relevant_count]) if row else 0.0
                irrelevant = max(row[relevant_count:]) if len(row) > relevant_count else 0.0
                score = relevant - (0.35 * irrelevant)
                frame["local_prefilter_score"] = float(score)
                frame["local_prefilter_reason"] = "clip"
                scored.append(frame)

        kept = self._select_clip_prefilter_frames(scored, min_score, top_k)
        kept.sort(key=lambda f: f.get("seconds") or 0.0)
        return kept, {
            "clip_available": True,
            "clip_model": model_name,
            "clip_device": device,
            "clip_min_score": min_score,
            "clip_top_k": top_k,
        }

    def _select_clip_prefilter_frames(self, scored, min_score, top_k):
        scored = list(scored or [])
        if not scored:
            return []
        scored.sort(key=lambda f: (f.get("local_prefilter_score") or 0.0), reverse=True)
        if top_k <= 0:
            return [f for f in scored if (f.get("local_prefilter_score") or 0.0) >= min_score]

        bucket_seconds = get_float_setting("INSPECTOR_LOCAL_CLIP_BUCKET_SECONDS", 60.0, min_value=10.0, max_value=600.0)
        per_bucket = get_int_setting("INSPECTOR_LOCAL_CLIP_PER_BUCKET", 2, min_value=0, max_value=20)
        selected = []
        selected_ids = set()
        if per_bucket > 0:
            buckets = {}
            for frame in scored:
                bucket = int(float(frame.get("seconds") or 0.0) // bucket_seconds)
                buckets.setdefault(bucket, []).append(frame)
            for bucket in sorted(buckets):
                bucket_frames = sorted(
                    buckets[bucket],
                    key=lambda f: (f.get("local_prefilter_score") or 0.0),
                    reverse=True,
                )
                for frame in bucket_frames[:per_bucket]:
                    frame_id = id(frame)
                    if frame_id not in selected_ids:
                        selected.append(frame)
                        selected_ids.add(frame_id)

        for frame in scored:
            if len(selected) >= top_k:
                break
            frame_id = id(frame)
            if frame_id in selected_ids:
                continue
            selected.append(frame)
            selected_ids.add(frame_id)

        return [
            f for f in selected
            if (f.get("local_prefilter_score") or 0.0) >= min_score
        ]

    def local_prefilter_frames(self, frames, progress_cb=None):
        stats = {
            "enabled": is_truthy_setting(get_setting("INSPECTOR_LOCAL_PREFILTER_ENABLED", "1")),
            "input_count": len(frames or []),
            "dedupe_enabled": is_truthy_setting(get_setting("INSPECTOR_LOCAL_DEDUPE_ENABLED", "1")),
            "clip_enabled": is_truthy_setting(get_setting("INSPECTOR_LOCAL_CLIP_ENABLED", "0")),
        }
        if not frames or not stats["enabled"]:
            stats["output_count"] = len(frames or [])
            return frames, stats

        filtered = list(frames)
        if stats["dedupe_enabled"]:
            threshold = get_int_setting("INSPECTOR_LOCAL_DEDUPE_THRESHOLD", 6, min_value=0, max_value=32)
            filtered = dedupe_similar_frames(filtered, threshold=threshold)
            stats["dedupe_threshold"] = threshold
            stats["after_dedupe_count"] = len(filtered)
            if progress_cb:
                progress_cb(16, f"Local prefilter deduped frames: {stats['input_count']} → {len(filtered)}")

        if stats["clip_enabled"] and filtered:
            model_name = get_setting("INSPECTOR_LOCAL_CLIP_MODEL", "openai/clip-vit-base-patch32")
            try:
                min_score = float(get_setting("INSPECTOR_LOCAL_CLIP_MIN_SCORE", 0.10))
            except (TypeError, ValueError):
                min_score = 0.10
            top_k = get_int_setting("INSPECTOR_LOCAL_CLIP_TOP_K", 80, min_value=0, max_value=2000)
            batch_size = get_int_setting("INSPECTOR_LOCAL_CLIP_BATCH_SIZE", 8, min_value=1, max_value=32)
            filtered, clip_stats = self._clip_prefilter_frames(
                filtered, model_name, min_score, top_k, batch_size, progress_cb=progress_cb
            )
            stats.update(clip_stats)
            stats["after_clip_count"] = len(filtered)
            if progress_cb:
                progress_cb(
                    18,
                    "Local CLIP prefilter "
                    f"({model_name}, {clip_stats.get('clip_device')}) kept {len(filtered)}/{stats.get('after_dedupe_count', stats['input_count'])} frames "
                    f"(top_k={top_k}, min_score={min_score})"
                )

        stats["output_count"] = len(filtered)
        return filtered, stats

    def _compress_video_for_vlm(self, input_path, output_path, progress_cb=None):
        """Compress large video file to low bitrate low res MP4 without audio using FFMPEG."""
        if progress_cb:
            progress_cb(12, "Compressing video to low resolution for VLM analysis...")
            
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", "scale=320:-2",
            "-c:v", "libx264",
            "-b:v", "150k",
            "-an",
            "-preset", "fast",
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"FFMPEG compression failed: {res.stderr}")

    def _extract_single_frame_at_timestamp(self, video_path, target_sec, dest_path):
        container = av.open(str(video_path))
        if not container.streams.video:
            container.close()
            raise RuntimeError("No video stream found in the container.")
        video_stream = container.streams.video[0]
        start_pts = video_stream.start_time
        if start_pts is None:
            start_time_offset = 0.0
        else:
            start_time_offset = float(start_pts * video_stream.time_base)
            
        target_pts = int((start_time_offset + target_sec) / video_stream.time_base)
        container.seek(target_pts, stream=video_stream)
        
        frame_found = None
        last_decoded_frame = None
        try:
            for frame in container.decode(video=0):
                last_decoded_frame = frame
                frame_absolute_sec = float(frame.pts * video_stream.time_base)
                frame_relative_sec = frame_absolute_sec - start_time_offset
                if frame_relative_sec >= target_sec:
                    frame_found = frame
                    break
        except av.AVError:
            pass
            
        if not frame_found and last_decoded_frame:
            frame_found = last_decoded_frame
            
        if frame_found:
            img = frame_found.to_image()
            width, height = img.size
            if width > 1024:
                new_height = int(height * 1024 / width)
                img = img.resize((1024, new_height), LANCZOS)
            img.save(dest_path, "JPEG", quality=90)
        container.close()

    def _extract_detection_preview_frames(self, video_path, job_dir, prefix, center_sec, primary_frame_name=""):
        before = get_float_setting("INSPECTOR_FINE_PREVIEW_BEFORE_SECONDS", 4.0, min_value=0.0, max_value=30.0)
        after = get_float_setting("INSPECTOR_FINE_PREVIEW_AFTER_SECONDS", 80.0, min_value=0.0, max_value=180.0)
        step = get_float_setting("INSPECTOR_FINE_PREVIEW_STEP_SECONDS", 8.0, min_value=1.0, max_value=30.0)
        max_frames = get_int_setting("INSPECTOR_FINE_PREVIEW_MAX_FRAMES", 12, min_value=1, max_value=30)

        offsets = [-before, 0.0]
        cursor = step
        while cursor <= after + 0.001:
            offsets.append(cursor)
            cursor += step

        timestamps = unique_preserve_order(
            max(0.0, float(center_sec) + offset)
            for offset in offsets[: max_frames + 2]
        )[:max_frames]

        preview_frames = []
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix)).strip("_") or "preview"
        primary_abs = round(float(center_sec), 3)
        for abs_sec in timestamps:
            ts_str = format_seconds(abs_sec)
            if primary_frame_name and round(abs_sec, 3) == primary_abs:
                frame_name = primary_frame_name
            else:
                frame_name = f"{safe_prefix}_{ts_str.replace(':', '_')}.jpg"
                dest_path = job_dir / frame_name
                try:
                    self._extract_single_frame_at_timestamp(video_path, abs_sec, dest_path)
                except Exception as e:
                    print(f"Error extracting preview frame for {ts_str}: {e}")
                    continue
            preview_frames.append({
                "timestamp": ts_str,
                "seconds": abs_sec,
                "image_file": frame_name,
                "offset": round(abs_sec - float(center_sec), 3),
            })
        return preview_frames

    def _run_native_video_inspection(self, container_path, actual_video_path, job_id, api_key, progress_cb=None):
        if progress_cb:
            progress_cb(11, "Preparing video for VLM analysis...")
            
        model_name = get_setting("INSPECTOR_MODEL", INSPECTOR_RECOMMENDED_FINE_MODEL)
        
        inspections_dir = get_local_video_dir() / "inspections"
        job_dir = inspections_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory(prefix="video-native-") as temp_dir:
            compressed_path = Path(temp_dir) / "compressed.mp4"
            
            # 1. Compress
            self._compress_video_for_vlm(actual_video_path, compressed_path, progress_cb)
            
            if progress_cb:
                progress_cb(40, "Encoding compressed video to base64...")
                
            # 2. Base64 Encode
            with open(compressed_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode("utf-8")
                
            if progress_cb:
                progress_cb(45, "Sending video to VLM (this might take a minute)...")
                
            # 3. Call OpenRouter/VLM API
            prompt = (
                "你是一个专门做视频画面审核的视觉分析模型。现在给你一个完整视频文件。\n"
                "请找出视频中人的腿或脚穿着薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜（stockings / pantyhose / hosiery）的场景。优先召回，不要漏掉疑似场景。\n\n"
                "目标与排除规则：\n"
                "- 目标只包括薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜。\n"
                "- 排除黑丝袜/黑丝、白丝袜和彩色丝袜。\n"
                "- 排除厚的、不透明的打底裤、厚连裤袜、leggings、保暖裤、普通棉袜或运动袜。\n"
                "- 明确裸腿/裸脚、普通短袜、纯长裤/西裤要排除；但裤脚下能看到丝袜覆盖的脚背、脚底、脚踝时例外。\n\n"
                + BAREFOOT_STOCKING_RECALL_RULES +
                "肉色/肤色丝袜识别要点：\n"
                "- 肉色丝袜可能非常像裸皮肤，请重点看脚底、脚背、脚尖、脚踝、小腿表面是否有连续、均匀、略带尼龙光泽的薄层。\n"
                "- 正证据包括：脚尖缝线、加固脚尖、袜口/袜边、脚趾细节被柔化或隐藏、趾甲看不清、脚趾缝不清楚、脚底/脚背像被薄膜覆盖、半透明织物纹理、尼龙层光泽。\n"
                "- 特别关注脱鞋、室内光脚、趴下/跪下/弯腰露出脚底、裤脚下露出脚背/脚踝的镜头。这些经常是关键画面。\n\n"
                "confidence 取值规则：\n"
                "- 'high'：特写/近景，脚底、脚背、脚尖、袜口或丝袜纹理清楚，有具体丝袜证据。\n"
                "- 'medium'：中景，脚/腿疑似薄款肉色或灰色丝袜，细节不如特写但值得复核。\n"
                "- 'low'：远景、模糊、局部遮挡，或只是弱候选，但对人工复核仍有价值。\n\n"
                "start_time/end_time 要覆盖最清楚的脚部/腿部/丝袜证据时间段，不要只返回人物刚入镜或刚进门的时间。reason 必须用中文说明具体视觉证据。\n\n"
                "只返回严格 JSON，字段名必须保持英文，格式如下：\n"
                "{\n"
                "  \"detections\": [\n"
                "    {\n"
                "      \"start_time\": \"00:02:10\",\n"
                "      \"end_time\": \"00:02:15\",\n"
                "      \"confidence\": \"high\" | \"medium\" | \"low\",\n"
                "      \"reason\": \"中文描述：画面是特写/中景/远景，具体看到哪些丝袜或候选证据\"\n"
                "    }\n"
                "  ]\n"
                "}\n"
                "如果完全没有候选，返回：\n"
                "{\n"
                "  \"detections\": []\n"
                "}"
            )
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if self.openrouter_referer:
                headers["HTTP-Referer"] = self.openrouter_referer
            if self.openrouter_app_title:
                headers["X-Title"] = self.openrouter_app_title
                
            payload = {
                "model": self._normalize_model_name(model_name),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{video_data}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            url = f"{self._api_base_url().rstrip('/')}/chat/completions"
            response = post_openrouter_with_retry(url, headers, payload, timeout=180)
            res_json = response.json()
            
            content = res_json["choices"][0]["message"].get("content")
            data = self._extract_json(content)
                
            vlm_detections = data.get("detections", [])
            
            if progress_cb:
                progress_cb(85, f"Processing {len(vlm_detections)} native video detections...")
                
            visual_detections = []
            require_concrete_evidence = is_truthy_setting(get_setting("INSPECTOR_STRICT_EVIDENCE_FILTER", "0"))
            for idx, det in enumerate(vlm_detections, start=1):
                if not isinstance(det, dict):
                    continue
                if not should_keep_model_detection(det, require_concrete_evidence):
                    continue
                start_time_str = det.get("start_time") or "00:00:00"
                end_time_str = det.get("end_time") or "00:00:00"
                start_sec = timestamp_to_seconds(start_time_str)
                end_sec = timestamp_to_seconds(end_time_str)
                mid_sec = (start_sec + end_sec) / 2.0
                
                mid_ts_str = format_seconds(mid_sec)
                
                frame_name = f"frame_native_{idx}_{mid_ts_str.replace(':', '_')}.jpg"
                dest_path = job_dir / frame_name
                
                try:
                    self._extract_single_frame_at_timestamp(actual_video_path, mid_sec, dest_path)
                except Exception as e:
                    print(f"Error extracting frame for {mid_ts_str}: {e}")
                    frame_name = ""
                    
                visual_detections.append({
                    "frame_id": f"frame_native_{idx}",
                    "timestamp": mid_ts_str,
                    "confidence": det.get("confidence") or "medium",
                    "reason": det.get("reason") or "",
                    "image_file": frame_name,
                    "seconds": mid_sec
                })
                
            return visual_detections

    def _clip_and_compress_segment(self, input_path, output_path, start_sec, duration):
        """Clip a short segment from the video and compress it using FFMPEG without audio."""
        width = get_int_setting("INSPECTOR_FINE_VIDEO_WIDTH", 768, min_value=320, max_value=1280)
        bitrate_k = get_int_setting("INSPECTOR_FINE_VIDEO_BITRATE_K", 650, min_value=200, max_value=2000)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(input_path),
            "-t", str(duration),
            "-vf", f"scale={width}:-2",
            "-c:v", "libx264",
            "-b:v", f"{bitrate_k}k",
            "-an",
            "-preset", "fast",
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"FFMPEG clipping failed: {res.stderr}")

    def _probe_video_file(self, video_path):
        try:
            cmd = [
                "ffprobe",
                "-hide_banner",
                "-loglevel", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames,duration",
                "-of", "json",
                str(video_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
            if res.returncode != 0:
                return {"duration": 0.0, "frames": 0, "error": res.stderr.strip()}
            data = json.loads(res.stdout or "{}")
            stream = (data.get("streams") or [{}])[0]
            return {
                "duration": float(stream.get("duration") or 0.0),
                "frames": int(stream.get("nb_read_frames") or 0),
                "error": "",
            }
        except Exception as e:
            return {"duration": 0.0, "frames": 0, "error": str(e)}

    def _video_duration_seconds(self, video_path) -> float:
        try:
            container = av.open(str(video_path))
            try:
                if container.duration:
                    return float(container.duration / 1000000.0)
                if container.streams.video:
                    video_stream = container.streams.video[0]
                    if video_stream.duration and video_stream.time_base:
                        return float(video_stream.duration * video_stream.time_base)
            finally:
                container.close()
        except Exception:
            pass
        probe = self._probe_video_file(video_path)
        return float(probe.get("duration") or 0.0)

    def empty_coarse_fallback_intervals(self, video_path, coarse_interval: float) -> list[tuple[float, float]]:
        if not is_truthy_setting(get_setting("INSPECTOR_EMPTY_COARSE_FALLBACK", "0")):
            return []
        duration = self._video_duration_seconds(video_path)
        if duration <= 0:
            return []
        min_window = get_float_setting("INSPECTOR_EMPTY_COARSE_FALLBACK_MIN_SECONDS", 180.0, min_value=10.0, max_value=1800.0)
        window = max(float(coarse_interval or 0.0), min_window)
        return [(0.0, max(duration, window))]

    def keep_coarse_when_fine_empty(self) -> bool:
        return is_truthy_setting(get_setting("INSPECTOR_KEEP_COARSE_ON_FINE_EMPTY", "1"))

    def coarse_fallback_detections(self, coarse_detections: list[dict]) -> list[dict]:
        fallback = []
        for det in coarse_detections or []:
            item = dict(det)
            item["source"] = item.get("source") or "coarse_fallback"
            item["needs_review"] = True
            item["fine_empty_fallback"] = True
            fallback.append(item)
        return fallback

    def _split_fine_video_intervals(self, intervals):
        max_duration = get_float_setting("INSPECTOR_FINE_VIDEO_MAX_SEGMENT_SECONDS", 45.0, min_value=5.0, max_value=180.0)
        max_segments = get_int_setting("INSPECTOR_FINE_VIDEO_MAX_SEGMENTS", 16, min_value=0, max_value=200)
        split = []
        for start, end in intervals:
            start = float(start)
            end = float(end)
            if end <= start:
                continue
            cursor = start
            while cursor < end:
                part_end = min(end, cursor + max_duration)
                split.append((cursor, part_end))
                cursor = part_end
        if max_segments and len(split) > max_segments:
            if max_segments == 1:
                return [split[0]]
            last = len(split) - 1
            indexes = sorted({
                round(i * last / (max_segments - 1))
                for i in range(max_segments)
            })
            selected = [split[i] for i in indexes]
            fill_idx = 0
            selected_set = set(indexes)
            while len(selected) < max_segments and fill_idx < len(split):
                if fill_idx not in selected_set:
                    selected.append(split[fill_idx])
                    selected_set.add(fill_idx)
                fill_idx += 1
            selected.sort(key=lambda item: item[0])
            return selected
        return split

    def _run_hybrid_video_fine_pass(self, actual_video_path, job_id, api_key, merged_intervals, fine_model_name, progress_cb=None):
        inspections_dir = get_local_video_dir() / "inspections"
        job_dir = inspections_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        extreme_recall = is_truthy_setting(get_setting("INSPECTOR_EXTREME_RECALL", "0"))
        exclude_male = is_truthy_setting(get_setting("INSPECTOR_EXCLUDE_MALE_SUBJECT", "1"))

        fine_detections = []
        fine_intervals = self._split_fine_video_intervals(merged_intervals)
        total_intervals = len(fine_intervals)
        
        for idx, (start, end) in enumerate(fine_intervals, start=1):
            if progress_cb:
                completed_pct = 60 + int((idx / total_intervals) * 35)
                progress_cb(completed_pct, f"Analyzing fine video segment {idx}/{total_intervals}...")
                
            duration = end - start
            with tempfile.TemporaryDirectory(prefix="video-clip-") as temp_dir:
                clip_path = Path(temp_dir) / f"clip_{idx}.mp4"
                
                try:
                    self._clip_and_compress_segment(actual_video_path, clip_path, start, duration)
                except Exception as e:
                    print(f"Error clipping segment {start}-{end}: {e}")
                    continue
                clip_probe = self._probe_video_file(clip_path)
                min_duration = get_float_setting("INSPECTOR_FINE_VIDEO_MIN_DURATION_SECONDS", 1.0, min_value=0.1, max_value=10.0)
                min_frames = get_int_setting("INSPECTOR_FINE_VIDEO_MIN_FRAMES", 5, min_value=1, max_value=300)
                if clip_probe["duration"] < min_duration or clip_probe["frames"] < min_frames:
                    self._append_trace_event(job_dir, {
                        "stage": "fine_video",
                        "kind": "video_segment",
                        "model": fine_model_name,
                        "trace_name": f"fine_segment_{idx}",
                        "start_seconds": start,
                        "end_seconds": end,
                        "duration": duration,
                        "clip_probe": clip_probe,
                        "skipped": True,
                        "skip_reason": "clip too short for video API",
                        "parsed_detections": [],
                    })
                    if progress_cb:
                        progress_cb(
                            completed_pct,
                            f"Fine {fine_model_name}: segment {idx}/{total_intervals} ({start:.1f}-{end:.1f}s) skipped; clip too short ({clip_probe['frames']} frame(s), {clip_probe['duration']:.2f}s)"
                        )
                    continue
                trace_clip_rel = None
                try:
                    trace_clip_dir = job_dir / "debug" / "fine_segments"
                    trace_clip_dir.mkdir(parents=True, exist_ok=True)
                    trace_clip_name = f"segment_{idx}_{format_seconds(start).replace(':', '_')}_{format_seconds(end).replace(':', '_')}.mp4"
                    trace_clip_path = trace_clip_dir / trace_clip_name
                    shutil.copy2(clip_path, trace_clip_path)
                    trace_clip_rel = str(trace_clip_path.relative_to(job_dir))
                except Exception as e:
                    print(f"Error saving trace clip {idx}: {e}")
                    
                # Base64 Encode
                try:
                    with open(clip_path, "rb") as f:
                        video_data = base64.b64encode(f.read()).decode("utf-8")
                except Exception as e:
                    print(f"Error encoding clip {idx}: {e}")
                    continue
                    
                prompt = (
                    "你是一个专门做视频画面审核的视觉分析模型。现在给你一个短视频片段。\n"
                    + (
                        "请按“高召回候选搜索”来分析：你的任务不是做最终判定，而是尽量找出值得人工复核的候选镜头。只要画面里有女性/女孩的裙装、制服、职业装、可见小腿/脚踝/脚/脚底、脱鞋、光脚样子、室内赤脚样子，或疑似薄款肉色/灰色丝袜/裤袜/连裤袜，都要返回候选。不要要求必须有丝袜铁证，宁可误报也不要漏掉。\n\n"
                        if extreme_recall else
                        "请分析视频中是否有人的腿或脚穿着薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜（stockings / pantyhose / hosiery）。\n\n"
                    ) +
                    "目标与排除规则：\n"
                    + ("- 高召回模式：腿、脚、脚底、脚踝、裙装、脱鞋、光脚样子、疑似肉色/灰色丝袜都作为候选返回，由用户人工剔除误报。\n" if extreme_recall else "- 只关注薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜。\n") +
                    "- 排除黑丝袜/黑丝、白丝袜和彩色丝袜。\n"
                    "- 排除厚的、不透明的打底裤、厚连裤袜、leggings、保暖裤、普通棉袜或运动袜。\n"
                    "- 只有当能清楚看到裸露脚趾/趾甲/自然皮肤纹理时，才排除为明确裸脚。脱鞋/看似光脚但可能穿肉色丝袜的镜头不要轻易排除。\n"
                    "- 普通长裤/西裤本身不是命中；但裤脚下露出被薄丝袜覆盖的脚背、脚底、脚踝时要返回。\n\n" +
                    BAREFOOT_STOCKING_RECALL_RULES +
                    "肉色/肤色丝袜识别要点：\n"
                    "- 肉色丝袜可能非常像裸皮肤，请重点看：脚底、脚背、脚尖、脚踝、小腿表面是否有连续、均匀、略带尼龙光泽的薄层。\n"
                    "- 正证据包括：脚尖缝线、加固脚尖、袜口/袜边、脚趾细节被柔化或隐藏、趾甲看不清、脚趾缝不清楚、脚底/脚背像被薄膜覆盖、半透明织物纹理、尼龙层光泽。\n"
                    "- 如果人物刚进门/上楼时还没露出关键脚部，但随后趴下、弯腰、脱鞋、脚底露出，请返回最清楚的后续时间点。\n\n"
                    "confidence 取值规则：\n"
                    "- 'high'：特写/近景，脚底、脚背、脚尖、袜口或丝袜纹理清楚，有具体丝袜证据。\n"
                    "- 'medium'：中景，脚/腿疑似薄款肉色或灰色丝袜，细节不如特写但值得复核。\n"
                    "- 'low'：远景、模糊、局部遮挡，或只是弱候选，但对人工复核仍有价值。\n\n"
                    "请返回 high、medium 和有用的 low 候选。细检阶段优先召回，不要为了精确而漏掉。reason 必须用中文，明确写出你看到的证据或候选理由，例如：脱鞋在旁、脚底露出、脚趾/趾甲被柔化、没有清楚脚趾缝、连续肉色薄层、袜口/缝线/尼龙纹理等。\n"
                    "time 字段必须返回“脚/脚底/脚背/脚尖/丝袜证据最清楚的时刻”，不要返回人物刚入镜、刚上楼、刚进门的时间。如果最佳画面是一小段，可以返回 'MM:SS-MM:SS'。\n\n"
                    "只返回严格 JSON，字段名必须保持英文，格式如下：\n"
                    "{\n"
                    "  \"detections\": [\n"
                    "    {\n"
                    "      \"time\": \"00:05\",\n"
                    "      \"confidence\": \"high\" | \"medium\" | \"low\",\n"
                    "      \"reason\": \"中文描述：画面是特写/中景/远景，具体看到哪些丝袜或候选证据\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "如果完全没有候选，返回：\n"
                    "{\n"
                    "  \"detections\": []\n"
                    "}"
                )
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                if self.openrouter_referer:
                    headers["HTTP-Referer"] = self.openrouter_referer
                if self.openrouter_app_title:
                    headers["X-Title"] = self.openrouter_app_title
                    
                payload = {
                    "model": self._normalize_model_name(fine_model_name),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "video_url",
                                    "video_url": {
                                        "url": f"data:video/mp4;base64,{video_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
                
                try:
                    url = f"{self._api_base_url().rstrip('/')}/chat/completions"
                    response = post_openrouter_with_retry(url, headers, payload, timeout=90)
                    res_json = response.json()
                    
                    content = res_json["choices"][0]["message"].get("content")
                    data = self._extract_json(content)
                        
                    clip_detections = data.get("detections", [])
                    self._append_trace_event(job_dir, {
                        "stage": "fine_video",
                        "kind": "video_segment",
                        "model": fine_model_name,
                        "trace_name": f"fine_segment_{idx}",
                        "start_seconds": start,
                        "end_seconds": end,
                        "duration": duration,
                        "clip_artifact": trace_clip_rel,
                        "raw_content": content,
                        "parsed_detections": clip_detections,
                    })
                    raw_count = len(clip_detections)
                    kept_count = 0
                    require_concrete_evidence = is_truthy_setting(get_setting("INSPECTOR_STRICT_EVIDENCE_FILTER", "0"))
                    keep_low = is_truthy_setting(get_setting("INSPECTOR_KEEP_LOW_FINE", "1"))
                    rep_offset = get_float_setting("INSPECTOR_FINE_VIDEO_REPRESENTATIVE_OFFSET_SECONDS", 4.0, min_value=0.0, max_value=20.0)
                    for c_det in clip_detections:
                        if not isinstance(c_det, dict):
                            continue
                        if not keep_fine_detection(c_det, extreme_recall=extreme_recall, require_concrete_evidence=require_concrete_evidence, keep_low_fine=keep_low, exclude_male_subject=exclude_male):
                            continue
                        raw_conf = c_det.get("confidence") or "low"

                        time_str = c_det.get("time") or "00:00"
                        abs_sec, source_abs_sec = fine_video_detection_seconds(time_str, start, duration, rep_offset)

                        abs_ts_str = format_seconds(abs_sec)
                        
                        frame_name = f"frame_fine_{idx}_{abs_ts_str.replace(':', '_')}.jpg"
                        dest_path = job_dir / frame_name
                        
                        try:
                            self._extract_single_frame_at_timestamp(actual_video_path, abs_sec, dest_path)
                        except Exception as e:
                            print(f"Error extracting frame for {abs_ts_str}: {e}")
                            frame_name = ""

                        preview_frames = []
                        if frame_name:
                            preview_frames = self._extract_detection_preview_frames(
                                actual_video_path,
                                job_dir,
                                f"frame_fine_{idx}_preview",
                                abs_sec,
                                primary_frame_name=frame_name,
                            )

                        fine_detections.append({
                            "frame_id": f"frame_fine_{idx}",
                            "timestamp": abs_ts_str,
                            "confidence": raw_conf,
                            "reason": c_det.get("reason") or "",
                            "image_file": frame_name,
                            "preview_frames": preview_frames,
                            "seconds": abs_sec,
                            "model_time": time_str,
                            "model_seconds": source_abs_sec,
                        })
                        kept_count += 1
                    if progress_cb:
                        progress_cb(
                            completed_pct,
                            f"Fine {fine_model_name}: segment {idx}/{total_intervals} ({start:.1f}-{end:.1f}s) returned {raw_count} raw, kept {kept_count}; total detections {len(fine_detections)}"
                        )
                except Exception as e:
                    message = str(e)
                    if is_quota_limit_error(message):
                        self._append_trace_event(job_dir, {
                            "stage": "fine_video",
                            "kind": "quota_stop",
                            "model": fine_model_name,
                            "trace_name": f"fine_segment_{idx}",
                            "start_seconds": start,
                            "end_seconds": end,
                            "duration": duration,
                            "clip_artifact": trace_clip_rel,
                            "clip_probe": clip_probe,
                            "stopped": True,
                            "stop_reason": message,
                            "detections_before_stop": len(fine_detections),
                            "parsed_detections": [],
                        })
                        if progress_cb:
                            progress_cb(
                                completed_pct,
                                f"Fine {fine_model_name}: quota/token limit at segment {idx}/{total_intervals}; stopping fine pass and keeping {len(fine_detections)} partial detection(s)"
                            )
                        break
                    if is_skippable_provider_content_error(message):
                        self._append_trace_event(job_dir, {
                            "stage": "fine_video",
                            "kind": "video_segment",
                            "model": fine_model_name,
                            "trace_name": f"fine_segment_{idx}",
                            "start_seconds": start,
                            "end_seconds": end,
                            "duration": duration,
                            "clip_artifact": trace_clip_rel,
                            "clip_probe": clip_probe,
                            "skipped": True,
                            "skip_reason": message,
                            "parsed_detections": [],
                        })
                        if progress_cb:
                            progress_cb(
                                completed_pct,
                                f"Fine {fine_model_name}: segment {idx}/{total_intervals} ({start:.1f}-{end:.1f}s) skipped after provider content inspection rejected the clip"
                            )
                        continue
                    if "Invalid video file" in message or "maximum allowed" in message or "StreamReadConstraints.getMaxStringLength" in message:
                        self._append_trace_event(job_dir, {
                            "stage": "fine_video",
                            "kind": "video_segment",
                            "model": fine_model_name,
                            "trace_name": f"fine_segment_{idx}",
                            "start_seconds": start,
                            "end_seconds": end,
                            "duration": duration,
                            "clip_artifact": trace_clip_rel,
                            "clip_probe": clip_probe,
                            "skipped": True,
                            "skip_reason": message,
                            "parsed_detections": [],
                        })
                        if progress_cb:
                            progress_cb(
                                completed_pct,
                                f"Fine {fine_model_name}: segment {idx}/{total_intervals} ({start:.1f}-{end:.1f}s) skipped after API rejected clip: {message[:160]}"
                            )
                        continue
                    raise RuntimeError(f"Error inspecting fine video segment {idx} ({start:.1f}-{end:.1f}s): {e}") from e
                    
        return fine_detections

    def _extract_frames_at_timestamps(self, video_path, temp_dir, timestamps, prefix):
        if not timestamps:
            return []

        container = av.open(str(video_path))
        if not container.streams.video:
            container.close()
            raise RuntimeError("No video stream found in the container.")
        
        video_stream = container.streams.video[0]
        start_pts = video_stream.start_time
        if start_pts is None:
            start_time_offset = 0.0
        else:
            start_time_offset = float(start_pts * video_stream.time_base)

        frames = []
        for idx, target_relative in enumerate(timestamps, start=1):
            target_absolute_sec = start_time_offset + target_relative
            target_pts = int(target_absolute_sec / video_stream.time_base)

            container.seek(target_pts, stream=video_stream)

            frame_found = None
            last_decoded_frame = None
            try:
                for frame in container.decode(video=0):
                    last_decoded_frame = frame
                    frame_absolute_sec = float(frame.pts * video_stream.time_base)
                    frame_relative_sec = frame_absolute_sec - start_time_offset
                    if frame_relative_sec >= target_relative:
                        frame_found = frame
                        break
            except av.AVError:
                pass

            if not frame_found and last_decoded_frame:
                frame_found = last_decoded_frame

            if frame_found:
                img = frame_found.to_image()
                width, height = img.size
                if width > 1024:
                    new_height = int(height * 1024 / width)
                    img = img.resize((1024, new_height), LANCZOS)
                
                file_name = f"{prefix}_{idx:04d}.jpg"
                file_path = Path(temp_dir) / file_name
                img.save(file_path, "JPEG", quality=90)
                
                frames.append({
                    "file_path": file_path,
                    "seconds": target_relative,
                    "timestamp_str": format_seconds(target_relative),
                    "id": f"{prefix}_{idx:04d}"
                })
        container.close()
        return frames

    def extract_frames(self, video_path, temp_dir, interval):
        # Fast path: a single linear ffmpeg decode (`fps=1/interval`) instead of
        # ~thousands of individual PyAV seeks, which is pathological on long
        # interlaced MPEG2 transport streams (each seek re-decodes a whole GOP).
        try:
            frames = self._extract_frames_ffmpeg(video_path, temp_dir, interval)
            if frames:
                return frames
        except Exception as e:
            print(f"ffmpeg frame extraction failed, falling back to PyAV: {e}")

        # Fallback: per-timestamp PyAV seeking (slow but robust).
        duration = self._video_duration_seconds(video_path)
        if duration <= 0:
            duration = 3600.0
        timestamps = []
        t = interval / 2.0
        while t < duration:
            timestamps.append(t)
            t += interval
        return self._extract_frames_at_timestamps(video_path, temp_dir, timestamps, "frame")

    def _run_ffmpeg_fps(self, video_path, temp_dir, interval, keyframe_only):
        """One ffmpeg pass: one frame per `interval`s, scaled to <=1024 wide.

        ``keyframe_only`` (``-skip_frame nokey``) decodes only I-frames — ~30x
        faster on long MPEG2 streams — and is accurate as long as keyframes are
        denser than `interval` (true for broadcast TV). Returns the JPEG paths.
        """
        out_pattern = str(Path(temp_dir) / "frame_%05d.jpg")
        vf = f"fps=1/{interval},scale='min(1024,iw)':-2"
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
        if keyframe_only:
            cmd += ["-skip_frame", "nokey"]
        cmd += ["-i", str(video_path), "-vf", vf, "-q:v", "3", out_pattern]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg fps extraction failed: {res.stderr[-300:]}")
        return sorted(Path(temp_dir).glob("frame_*.jpg"))

    def _extract_frames_ffmpeg(self, video_path, temp_dir, interval):
        """Extract one frame per `interval` seconds in a single ffmpeg pass.

        Output frame i corresponds to playback time ~= i*interval (the same
        0-based timeline mpv --start and the manual marks use). Tries the fast
        keyframe-only decode first; if it yields too few frames (sparse
        keyframes would misalign the i*interval timing) it falls back to a full
        decode for correctness.
        """
        interval = max(0.5, float(interval))
        duration = self._video_duration_seconds(video_path)
        expected = (duration / interval) if duration > 0 else 0

        files = self._run_ffmpeg_fps(video_path, temp_dir, interval, keyframe_only=True)
        if expected and len(files) < 0.9 * expected:
            # Keyframes too sparse for accurate i*interval timing -> full decode.
            for f in files:
                try:
                    f.unlink()
                except OSError:
                    pass
            files = self._run_ffmpeg_fps(video_path, temp_dir, interval, keyframe_only=False)

        frames = []
        for i, fp in enumerate(files):
            sec = round(i * interval, 3)
            frames.append({
                "file_path": fp,
                "seconds": float(sec),
                "timestamp_str": format_seconds(sec),
                "id": f"frame_{i + 1:04d}",
            })
        return frames

    def merge_intervals(self, intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = current
            if curr_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(current)
        return merged

    def fine_window_for_detection(self, detection: dict, coarse_interval: float) -> tuple[float, float]:
        t = float(detection.get("seconds") or 0.0)
        conf = str(detection.get("confidence") or "low").strip().lower()
        default_backward = max(
            coarse_interval / 2.0,
            get_float_setting("INSPECTOR_FINE_BACKWARD_PADDING_SECONDS", 8.0, min_value=0.5, max_value=60.0),
        )
        default_forward = max(
            coarse_interval / 2.0,
            get_float_setting("INSPECTOR_FINE_FORWARD_PADDING_SECONDS", 90.0, min_value=0.5, max_value=180.0),
        )
        legacy_padding = get_float_setting("INSPECTOR_FINE_PADDING_SECONDS", 0.0, min_value=0.0, max_value=60.0)
        if legacy_padding > 0:
            default_backward = max(default_backward, legacy_padding)
            default_forward = max(default_forward, legacy_padding)
        low_padding = max(default_backward, default_forward, get_float_setting("INSPECTOR_FINE_LOW_PADDING_SECONDS", 24.0, min_value=1.0, max_value=120.0))
        reason = str(detection.get("reason") or "")
        is_far_or_weak = conf == "low" or any(term in reason for term in ("远景", "距离较远", "细节不清", "小腿", "脚踝"))
        backward = low_padding if is_far_or_weak else default_backward
        forward = low_padding if is_far_or_weak else default_forward
        return max(0.0, t - backward), t + forward

    def extract_frames_range(self, video_path, temp_dir, start_time, end_time, interval, prefix="frame_fine"):
        timestamps = []
        t = start_time + interval / 2.0
        while t < end_time:
            timestamps.append(t)
            t += interval
        if not timestamps:
            timestamps.append((start_time + end_time) / 2.0)

        return self._extract_frames_at_timestamps(video_path, temp_dir, timestamps, prefix)

    def resize_and_encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def inspect_batch(self, batch, api_key, model_name, is_coarse=False, job_dir=None, trace_name=None, trace_stage=None, prompt_override=None):
        if not api_key:
            raise RuntimeError(f"{self._api_provider().upper()} API key is not set. Cannot run visual analysis.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_referer:
            headers["HTTP-Referer"] = self.openrouter_referer
        if self.openrouter_app_title:
            headers["X-Title"] = self.openrouter_app_title

        extreme_recall = is_truthy_setting(get_setting("INSPECTOR_EXTREME_RECALL", "0"))
        is_fine_screenshot = (not is_coarse) and str(trace_stage or "") == "screenshot_fine"
        if extreme_recall:
            inclusion_rule = (
                "关键召回规则（高召回候选搜索）：\n"
                "- 你的任务不是最终判定丝袜，而是为人工复核收集候选帧。\n"
                "- 只要画面出现女性/女孩的可见腿、小腿、脚踝、脚、脚底、裙装、连衣裙、制服、职业装、脱鞋、室内光脚样子、鞋在旁边、或疑似薄款肉色/灰色丝袜/裤袜，都要返回。\n"
                "- 远景、模糊、局部遮挡、不确定场景也返回，confidence='low'。\n"
                "- 即使可能是裸脚也要返回候选；用户宁可看到误报，也不希望漏掉。\n"
                "- 只有完全没有人/下半身、纯男性画面、只有脸/上半身、文字画面、风景、食物物品，或明确长裤且没有脚/踝/腿可见时，才排除。\n\n"
            )
        elif is_coarse:
            inclusion_rule = (
                "关键召回规则（粗检）：\n"
                "- 这是粗检候选选择，不要做过严最终判断。只要可能包含薄款肉色/灰色丝袜/裤袜，或有值得后续视频细检的腿脚场景，就返回 medium 或 low。\n"
                "- 只有确定是误报时才排除，例如明确裸脚且脚趾/趾甲清楚、普通白袜/黑袜/运动袜、牛仔裤/厚衣物、没有腿脚可见。\n"
                "- 女性穿裙装/连衣裙/职业装/制服，露出小腿/脚踝/脚，或脱鞋/室内光脚样子，即使远景、全身、脚很小、细节不清，也作为 low 候选返回。\n"
                "- 光脚/脱鞋镜头只有在能清楚看到裸露脚趾/趾甲/自然皮肤纹理时，才排除为裸脚；否则按候选返回。\n"
                "- 长裤/西裤场景只有在裤脚下可见疑似薄丝袜覆盖的脚背/脚底/脚踝时返回；黑色长裤下明显裸脚通常排除。\n\n"
            )
        elif is_fine_screenshot:
            inclusion_rule = (
            "关键召回规则（截图细检）：\n"
            "- 这些帧来自粗检命中的候选场景，请优先召回。\n"
            "- 只要视觉线索支持薄款肉色/灰色丝袜/裤袜，就返回 high、medium 或有用的 low。\n"
            "- 光脚/脱鞋/室内赤脚样子的裙装、职业装、制服场景，不要求必须看到脚尖缝线。脚/踝看起来被均匀薄层覆盖、脚趾/趾甲被柔化或隐藏、鞋在旁边、脚底/脚背像有连续肉色丝袜层，都要返回。\n"
            "- 只排除明确误报：清楚裸露脚趾/趾甲/自然皮肤纹理、普通袜、长裤下明显裸脚、厚不透明打底裤/leggings、黑/白/彩色丝袜，或腿脚区域不可见。\n\n"
            )
        else:
            inclusion_rule = (
                "关键规则（普通精确模式）：\n"
                "- 只有当你确定或高度怀疑画面包含目标丝袜/裤袜时，才放入 detections。\n"
                "- 裸腿/裸脚、普通袜、长裤下没有丝袜证据、或完全无法判断时，不要返回。\n"
                "- 光脚/脱鞋场景需要可见的丝袜证据；长裤/西裤场景需要裤脚下可见具体丝袜覆盖证据。\n\n"
            )

        use_simple_coarse = (
            is_coarse and not extreme_recall
            and is_truthy_setting(get_setting("INSPECTOR_SIMPLE_COARSE_PROMPT", "1"))
        )
        prompt = SIMPLE_COARSE_PROMPT if use_simple_coarse else (
            "你是一个专门做视频截图审核的视觉分析模型。下面是一组按时间顺序排列的视频帧，每帧都有 Frame ID 和 Timestamp。"
            + (
                "请按高召回候选搜索分析每张图：只要可能包含丝袜相关场景，或值得人工复核的女性腿脚/裙装/脱鞋/光脚样子/疑似肉色或灰色丝袜，都返回。不要要求必须有丝袜铁证。\n\n"
                if extreme_recall else
                "请判断每张图是否出现人的腿或脚穿着薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜（stockings / pantyhose / hosiery）。\n\n"
            ) +
            "目标与排除规则：\n"
            + ("- 高召回模式：腿、脚、脚底、脚踝、裙装、脱鞋、光脚样子、疑似肉色/灰色丝袜都作为候选，用户会人工剔除误报。\n" if extreme_recall else "- 只关注薄款、半透明、肉色/肤色或灰色丝袜/裤袜/连裤袜。\n") +
            "- 排除黑丝袜/黑丝、白丝袜和彩色丝袜。\n"
            "- 排除厚的、不透明的打底裤、厚连裤袜、leggings、保暖裤、普通棉袜或运动袜。\n"
            + ("" if extreme_recall else "- 排除明确裸腿/裸脚、普通短袜、纯长裤/西裤；但裤脚下能看到丝袜覆盖的脚/踝时例外。\n\n")
            + inclusion_rule +
            ("" if (is_coarse or extreme_recall) else STRICT_STOCKING_EVIDENCE_RULES) +
            (BAREFOOT_STOCKING_RECALL_RULES if (is_coarse or extreme_recall) else "") +
            "肉色/肤色丝袜识别要点：\n"
            "- 肉色丝袜可能很像裸皮肤，请重点看脚底、脚背、脚尖、脚踝、小腿是否有连续、均匀、略带尼龙光泽的薄层。\n"
            "- 正证据包括：脚尖缝线、加固脚尖、袜口/袜边、脚趾细节被柔化或隐藏、趾甲看不清、脚趾缝不清楚、脚底/脚背像被薄膜覆盖、半透明织物纹理、尼龙层光泽。\n"
            "- 远景/全身镜头仍可能有用；尤其是女性裙装/职业装/制服、室内脱鞋/光脚、趴下或弯腰露出脚底时，不要轻易丢弃。\n\n"
            "confidence 取值规则：\n"
            "- 'high'：特写/近景，脚底、脚背、脚尖、袜口或丝袜纹理清楚，有具体丝袜证据。\n"
            "- 'medium'：中景，脚/腿疑似薄款肉色或灰色丝袜，细节不如特写但值得复核。\n"
            "- 'low'：远景、模糊、局部遮挡，或只是弱候选，但对人工复核仍有价值。\n\n"
            + ("返回所有有用的 low 候选。宁可误报也不要漏掉，用户要看可能漏检的镜头。\n\n" if extreme_recall else ("" if is_coarse else "返回 high、medium 和有用的 low 候选。reason 必须用中文说明具体视觉线索，例如脚底露出、脚趾/趾甲被柔化、鞋在旁边、连续肉色薄层、袜口/缝线/尼龙纹理等。\n\n")) +
            "只返回严格 JSON，字段名必须保持英文，格式如下：\n"
            "{\n"
            "  \"detections\": [\n"
            "    {\n"
            "      \"frame_id\": \"frame_0012\",\n"
            "      \"timestamp\": \"00:02:00\",\n"
            "      \"confidence\": \"high\" | \"medium\" | \"low\",\n"
            "      \"reason\": \"中文描述：画面是特写/中景/远景，具体看到哪些丝袜或候选证据\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "如果完全没有候选，返回：\n"
            "{\n"
            "  \"detections\": []\n"
            "}"
        )

        # Tier-B replay can swap in a candidate prompt to A/B against the
        # committed corpus without re-extracting frames from the source video.
        if prompt_override:
            prompt = prompt_override

        content_list = [
            {"type": "text", "text": prompt}
        ]

        for frame in batch:
            try:
                base64_data = self.resize_and_encode_image(frame["file_path"])
                content_list.append({
                    "type": "text",
                    "text": f"Frame ID: {frame['id']} (Timestamp: {frame['timestamp_str']})"
                })
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_data}"
                    }
                })
            except Exception as e:
                print(f"Error encoding image {frame['file_path']}: {e}")

        payload = {
            "model": self._normalize_model_name(model_name),
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }

        url = f"{self._api_base_url().rstrip('/')}/chat/completions"
        try:
            response = post_openrouter_with_retry(url, headers, payload, timeout=90)
        except RuntimeError as e:
            if is_skippable_provider_content_error(e):
                if job_dir and trace_name:
                    self._append_trace_event(Path(job_dir), {
                        "stage": trace_stage or ("coarse" if is_coarse else "fine"),
                        "kind": "image_batch",
                        "model": model_name,
                        "trace_name": trace_name,
                        "frames": [self._copy_trace_frame(frame, Path(job_dir), trace_name) for frame in batch],
                        "skipped": True,
                        "skip_reason": str(e),
                        "parsed_detections": [],
                    })
                return []
            raise

        response_data = response.json()
        choices = response_data.get("choices") or []
        if not choices:
            if job_dir and trace_name:
                self._append_trace_event(Path(job_dir), {
                    "stage": trace_stage or ("coarse" if is_coarse else "fine"),
                    "kind": "image_batch",
                    "model": model_name,
                    "trace_name": trace_name,
                    "frames": [self._copy_trace_frame(frame, Path(job_dir), trace_name) for frame in batch],
                    "raw_response": response_data,
                    "parsed_detections": [],
                })
            return []
            
        content = choices[0].get("message", {}).get("content")
        parsed = self._extract_json(content)
        # Tolerate models that return a bare JSON array instead of {"detections": [...]}
        if isinstance(parsed, list):
            detections = parsed
        else:
            detections = parsed.get("detections", [])
        if job_dir and trace_name:
            self._append_trace_event(Path(job_dir), {
                "stage": trace_stage or ("coarse" if is_coarse else "fine"),
                "kind": "image_batch",
                "model": model_name,
                "trace_name": trace_name,
                "frames": [self._copy_trace_frame(frame, Path(job_dir), trace_name) for frame in batch],
                "prompt_summary": "chronological screenshot batch stockings/pantyhose detection",
                "raw_content": content,
                "parsed_detections": detections,
            })
        return detections

    def verify_detection_image(self, image_path, api_key, model_name, detection):
        if not api_key or not model_name or not image_path or not Path(image_path).exists():
            return {"keep": True, "reason": "未复核"}

        image_data = self.resize_and_encode_image(image_path)
        prompt = (
            "Look at this frame independently. Is the person visibly wearing thin nude/skin-colored or grey sheer stockings/pantyhose?\n"
            "Ignore the previous model unless the image itself supports it. If you are not sure, choose keep=false.\n"
            "Do not infer stockings from smooth skin, lighting, blur, or compression artifacts. Keep only when you can point to visible hosiery evidence in the image.\n\n"
            f"Previous candidate reason, which may be wrong: {detection.get('reason') or ''}\n\n"
            "Respond in strict JSON only:\n"
            "{\n"
            "  \"keep\": true | false,\n"
            "  \"confidence\": \"high\" | \"medium\" | \"low\",\n"
            "  \"visible_clothing\": \"Chinese description of visible clothing/footwear\",\n"
            "  \"visible_hosiery_evidence\": [\"what in the image proves stockings/pantyhose\"],\n"
            "  \"rejection_reason\": \"Chinese explanation when keep=false, otherwise empty string\",\n"
            "  \"reason\": \"Chinese final explanation\"\n"
            "}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.openrouter_referer:
            headers["HTTP-Referer"] = self.openrouter_referer
        if self.openrouter_app_title:
            headers["X-Title"] = self.openrouter_app_title

        payload = {
            "model": self._normalize_model_name(model_name),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        try:
            url = f"{self._api_base_url().rstrip('/')}/chat/completions"
            response = post_openrouter_with_retry(url, headers, payload, timeout=60)
            response_data = response.json()
            choices = response_data.get("choices") or []
            if not choices:
                return {"keep": False, "reason": "复核模型没有返回结果"}
            content = choices[0].get("message", {}).get("content")
            parsed = self._extract_json(content)
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(format_openrouter_error(e)) from e

        evidence = parsed.get("visible_hosiery_evidence")
        if not isinstance(evidence, list):
            evidence = []
        evidence = [str(item).strip() for item in evidence if str(item).strip()]
        keep = bool(parsed.get("keep")) and bool(evidence)

        return {
            "keep": keep,
            "confidence": str(parsed.get("confidence") or detection.get("confidence") or "medium").strip().lower(),
            "visible_clothing": str(parsed.get("visible_clothing") or "").strip(),
            "visible_hosiery_evidence": evidence,
            "rejection_reason": str(parsed.get("rejection_reason") or "").strip(),
            "reason": str(parsed.get("reason") or "").strip(),
        }

    def verify_visual_detections(self, detections, job_dir, api_key, model_name, progress_cb=None):
        if not detections or not api_key or not model_name:
            return detections

        policy = get_setting("INSPECTOR_VERIFY_POLICY", "medium")
        verified = []
        total = len(detections)
        for idx, det in enumerate(detections, start=1):
            if not should_verify_detection(det, policy):
                verified.append(det)
                continue
            if progress_cb:
                progress_cb(96, f"Final verifier: checking candidate {idx}/{total}...")
            image_file = det.get("image_file") or ""
            image_path = Path(job_dir) / image_file if image_file else None
            verdict = self.verify_detection_image(image_path, api_key, model_name, det)
            try:
                if image_path and Path(image_path).exists():
                    trace_frame = {
                        "id": det.get("frame_id") or f"verify_{idx}",
                        "timestamp_str": det.get("timestamp") or format_seconds(det.get("seconds") or 0),
                        "seconds": det.get("seconds"),
                        "file_path": Path(image_path),
                    }
                    self._append_trace_event(Path(job_dir), {
                        "stage": "verify",
                        "kind": "image_verify",
                        "model": model_name,
                        "trace_name": f"verify_{idx:03d}",
                        "frames": [self._copy_trace_frame(trace_frame, Path(job_dir), f"verify_{idx:03d}")],
                        "previous_reason": det.get("reason") or "",
                        "parsed_detections": [verdict],
                    })
            except Exception as e:
                print(f"Error saving verifier trace: {e}")
            det["verify_model"] = model_name
            det["verify_reason"] = verdict.get("reason") or ""
            det["visible_clothing"] = verdict.get("visible_clothing") or ""
            det["visible_hosiery_evidence"] = verdict.get("visible_hosiery_evidence") or []
            det["rejection_reason"] = verdict.get("rejection_reason") or ""
            if not verdict.get("keep") or not det["visible_hosiery_evidence"]:
                continue
            if verdict.get("confidence") in {"high", "medium", "low"}:
                det["confidence"] = verdict["confidence"]
            if verdict.get("reason"):
                det["reason"] = verdict["reason"]
            if is_false_positive_reason(det.get("reason", "")):
                continue
            verified.append(det)
        return verified

    def _extract_json(self, text):
        if text is None:
            return {"detections": []}
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                return {"detections": []}
        text = text.strip()
        if not text:
            return {"detections": []}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"detections": []}

    def scan_subtitles_for_keywords(self, container_path, host_video_path):
        # Look for subtitles next to host video path or next to local copy path
        srt_candidates = [
            host_video_path.with_suffix(".bilingual.srt"),
            host_video_path.with_suffix(".orig.srt"),
        ]
        
        local_path = get_local_copy_path(container_path)
        if local_path.exists():
            srt_candidates.insert(0, local_path.with_suffix(".bilingual.srt"))
            srt_candidates.insert(1, local_path.with_suffix(".orig.srt"))

        srt_path = None
        for candidate in srt_candidates:
            if candidate.exists():
                srt_path = candidate
                break
                
        if not srt_path:
            return []

        try:
            content = srt_path.read_text(encoding="utf-8")
        except Exception:
            return []
            
        blocks = re.split(r"\n\s*\n", content.strip(), flags=re.MULTILINE)
        
        keywords = [
            r"ストッキング", r"パンスト", r"タイツ", r"美脚", r"レギンス", r"シアー",
            r"丝袜", r"裤袜", r"连裤袜", r"袜脚"
        ]
        pattern = re.compile("|".join(keywords), re.IGNORECASE)
        
        matches = []
        for block in blocks:
            lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
            if len(lines) < 3:
                continue

            time_line = lines[1]
            if "-->" not in time_line:
                continue

            start_raw, _ = [part.strip() for part in time_line.split("-->", 1)]
            text = "\n".join(lines[2:]).strip()
            
            if pattern.search(text):
                try:
                    seconds = parse_srt_timestamp(start_raw)
                    matches.append({
                        "timestamp": format_seconds(seconds),
                        "text": text,
                        "seconds": seconds
                    })
                except Exception:
                    pass
        return matches

    def run_inspection(self, container_path, interval, job_id, progress_cb=None):
        host_video_path = Path(to_host_path(container_path))
        
        # Decide which video file path to run ffmpeg on: local copy preferred if it exists
        local_path = get_local_copy_path(container_path)
        if local_path.exists() and local_path.is_file() and local_path.stat().st_size > 0:
            actual_video_path = local_path
        else:
            actual_video_path = host_video_path

        if not actual_video_path.exists():
            raise FileNotFoundError(f"Video file not found: {actual_video_path}")

        inspections_dir = get_local_video_dir() / "inspections"
        job_dir = inspections_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(5, "Scanning subtitles for keywords...")
            
        subtitle_matches = self.scan_subtitles_for_keywords(container_path, host_video_path)

        coarse_interval = interval
        fine_interval = 2

        # If interval is already small, do a single pass to save time/calls
        is_two_pass = coarse_interval > fine_interval

        visual_detections = []
        coarse_detections = []  # also the temporal-corroboration support pool

        if progress_cb:
            progress_cb(10, f"Extracting coarse frames at {coarse_interval}s interval...")

        api_key = self._api_key()
        coarse_model_name = get_setting("INSPECTOR_MODEL", INSPECTOR_RECOMMENDED_COARSE_MODEL)
        fine_model_name = get_setting("INSPECTOR_FINE_MODEL", INSPECTOR_RECOMMENDED_FINE_MODEL)
        if not fine_model_name:
            fine_model_name = coarse_model_name
        verify_enabled = is_truthy_setting(get_setting("INSPECTOR_VERIFY_ENABLED", "0"))
        verify_model_name = get_setting("INSPECTOR_VERIFY_MODEL", INSPECTOR_RECOMMENDED_VERIFY_MODEL)
        coarse_batch_size = get_int_setting("INSPECTOR_COARSE_BATCH_SIZE", 10, min_value=1, max_value=20)
        fine_batch_size = get_int_setting("INSPECTOR_FINE_BATCH_SIZE", 6, min_value=1, max_value=12)
        require_concrete_evidence = is_truthy_setting(get_setting("INSPECTOR_STRICT_EVIDENCE_FILTER", "0"))
        extreme_recall = is_truthy_setting(get_setting("INSPECTOR_EXTREME_RECALL", "0"))
        exclude_male = is_truthy_setting(get_setting("INSPECTOR_EXCLUDE_MALE_SUBJECT", "1"))
        skip_fine_pass = is_truthy_setting(get_setting("INSPECTOR_SKIP_FINE_PASS", "0"))
        drop_weak_coarse = is_truthy_setting(get_setting("INSPECTOR_DROP_WEAK_COARSE", "0"))

        inspector_mode = get_setting("INSPECTOR_MODE", "hybrid_video")
        local_prefilter_stats = None

        if inspector_mode == "native_video":
            if progress_cb:
                progress_cb(9, f"Mode native_video: sending compressed full video to {coarse_model_name}; strict_evidence_filter={require_concrete_evidence}")
            visual_detections = self._run_native_video_inspection(
                container_path, actual_video_path, job_id, api_key, progress_cb
            )
        elif inspector_mode == "hybrid_video":
            coarse_detections = []
            with tempfile.TemporaryDirectory(prefix="video-inspect-") as temp_dir:
                coarse_frames = self.extract_frames(actual_video_path, temp_dir, coarse_interval)
                coarse_frames, local_prefilter_stats = self.local_prefilter_frames(coarse_frames, progress_cb=progress_cb)
                total_coarse = len(coarse_frames)
                if total_coarse > 0:
                    if progress_cb:
                        progress_cb(20, f"Coarse pass: {coarse_model_name} will inspect {total_coarse} prefiltered frames in batches of {coarse_batch_size}")
                    coarse_batches = [coarse_frames[i:i + coarse_batch_size] for i in range(0, total_coarse, coarse_batch_size)]
                    coarse_concurrency = get_int_setting("INSPECTOR_COARSE_CONCURRENCY", 6, min_value=1, max_value=16)
                    if progress_cb:
                        progress_cb(20, f"Coarse pass: {coarse_model_name} inspecting {total_coarse} frames in {len(coarse_batches)} batch(es), {coarse_concurrency}x parallel")

                    # Run the (slow, network-bound) batch API calls concurrently;
                    # process each result in order as it arrives. Trace writes are
                    # serialized by self._trace_lock, frame-copy/append run here on
                    # the main thread, so the shared state stays consistent.
                    # A single batch failing (e.g. a transient provider 400 under
                    # concurrency) must NOT nuke a multi-hour inspection: catch
                    # per-batch, skip the failed one, and continue. Only abort if
                    # EVERY batch failed (a real config/key/model problem).
                    def _run_coarse_batch(item):
                        b_idx, batch = item
                        try:
                            dets = self.inspect_batch(
                                batch, api_key, coarse_model_name, is_coarse=True,
                                job_dir=job_dir, trace_name=f"coarse_batch_{b_idx + 1:03d}", trace_stage="coarse_screenshot",
                            )
                            return b_idx, batch, dets, None
                        except Exception as e:
                            return b_idx, batch, [], e

                    coarse_failures = []
                    last_coarse_error = None
                    with ThreadPoolExecutor(max_workers=coarse_concurrency) as _ex:
                        for done, (b_idx, batch, detections, err) in enumerate(
                            _ex.map(_run_coarse_batch, list(enumerate(coarse_batches))), start=1
                        ):
                            if err is not None:
                                coarse_failures.append(b_idx)
                                last_coarse_error = err
                                print(f"Coarse batch {b_idx + 1}/{len(coarse_batches)} failed, skipped: {err}")
                                if progress_cb:
                                    progress_cb(
                                        20 + int((done / len(coarse_batches)) * 30),
                                        f"Coarse {coarse_model_name}: {done}/{len(coarse_batches)} done; {len(coarse_failures)} batch(es) skipped; {len(coarse_detections)} candidates"
                                    )
                                continue
                            kept_in_batch = 0
                            for det in detections:
                                if not keep_coarse_detection(det, extreme_recall=extreme_recall, hybrid=True, is_two_pass=is_two_pass, exclude_male_subject=exclude_male, drop_weak_coarse=drop_weak_coarse):
                                    continue
                                frame_id = det.get("frame_id")
                                matched_frame = next((f for f in batch if f["id"] == frame_id), None)
                                if matched_frame:
                                    dest_name = f"{matched_frame['id']}_{matched_frame['timestamp_str'].replace(':', '_')}.jpg"
                                    dest_path = job_dir / dest_name
                                    shutil.copy2(matched_frame["file_path"], dest_path)

                                    det["image_file"] = dest_name
                                    det["seconds"] = matched_frame["seconds"]
                                    det["timestamp"] = matched_frame["timestamp_str"]
                                    coarse_detections.append(det)
                                    kept_in_batch += 1
                            if progress_cb:
                                progress_cb(
                                    20 + int((done / len(coarse_batches)) * 30),
                                    f"Coarse {coarse_model_name}: {done}/{len(coarse_batches)} batches done; {len(coarse_detections)} candidates so far"
                                )

                    if coarse_failures and len(coarse_failures) == len(coarse_batches):
                        # Every batch failed -> not transient; surface the real error.
                        raise RuntimeError(f"Error inspecting coarse frame batches: {last_coarse_error}") from last_coarse_error
                    if coarse_failures:
                        print(f"Coarse pass: {len(coarse_failures)}/{len(coarse_batches)} batch(es) failed and were skipped (transient); continuing with {len(coarse_detections)} candidates")
                            
            if skip_fine_pass:
                # Coarse-only mode: the (filtered) coarse candidates ARE the
                # result. On the current ground truth this beats the fine-video
                # pass on both recall and precision — the fine stage processes
                # only a fraction of candidates and confirms the wrong ones.
                visual_detections = coarse_detections
                # NOTE: confidence is an unreliable signal here — real faint scenes
                # come back "low" while smooth-skin false positives come back "high".
                # So confidence gating is OFF by default; the semantic reason filter
                # is the real judge. Opt back in with INSPECTOR_COARSE_DROP_LOW=1.
                if is_truthy_setting(get_setting("INSPECTOR_COARSE_DROP_LOW", "0")):
                    visual_detections = [
                        d for d in visual_detections
                        if str(d.get("confidence", "")).strip().lower() != "low"
                    ]
                if progress_cb:
                    progress_cb(90, f"Coarse-only mode (skip fine): {len(visual_detections)} candidate(s) kept for review")
            elif coarse_detections or subtitle_matches:
                if progress_cb:
                    progress_cb(55, f"Fine pass setup: {len(coarse_detections)} visual candidates + {len(subtitle_matches)} subtitle hits")
                intervals = []
                for det in coarse_detections:
                    start, end = self.fine_window_for_detection(det, coarse_interval)
                    intervals.append((start, end))
                for match in subtitle_matches:
                    t = match.get("seconds")
                    if t is not None:
                        start = max(0.0, t - (coarse_interval / 2.0))
                        end = t + (coarse_interval / 2.0)
                        intervals.append((start, end))
                merged_intervals = self.merge_intervals(intervals)
                if progress_cb:
                    progress_cb(58, f"Fine pass: {fine_model_name} will inspect {len(merged_intervals)} merged video segment(s)")
                
                visual_detections = self._run_hybrid_video_fine_pass(
                    actual_video_path, job_id, api_key, merged_intervals, fine_model_name, progress_cb
                )
                if not visual_detections and coarse_detections and self.keep_coarse_when_fine_empty():
                    visual_detections = self.coarse_fallback_detections(coarse_detections)
                    if progress_cb:
                        progress_cb(
                            90,
                            f"Fine pass kept 0 detections; preserving {len(visual_detections)} coarse candidate(s) for review"
                        )
            else:
                fallback_intervals = self.empty_coarse_fallback_intervals(actual_video_path, coarse_interval)
                if fallback_intervals:
                    if progress_cb:
                        progress_cb(
                            55,
                            "No coarse visual candidates or subtitle hits; running empty-coarse coverage fallback"
                        )
                    visual_detections = self._run_hybrid_video_fine_pass(
                        actual_video_path, job_id, api_key, fallback_intervals, fine_model_name, progress_cb
                    )
                else:
                    visual_detections = []
                    if progress_cb:
                        progress_cb(90, "No coarse visual candidates or subtitle keyword hits; skipping fine pass")
        else:
            with tempfile.TemporaryDirectory(prefix="video-inspect-") as temp_dir:
                coarse_frames = self.extract_frames(actual_video_path, temp_dir, coarse_interval)
                coarse_frames, local_prefilter_stats = self.local_prefilter_frames(coarse_frames, progress_cb=progress_cb)
                total_coarse = len(coarse_frames)
                
                if total_coarse == 0:
                    raise RuntimeError("No frames extracted from the video.")
                
                if progress_cb:
                    progress_cb(20, f"Screenshot coarse pass: {coarse_model_name} will inspect {total_coarse} prefiltered frames in batches of {coarse_batch_size}")
                    
                coarse_batches = [coarse_frames[i:i + coarse_batch_size] for i in range(0, total_coarse, coarse_batch_size)]
                coarse_detections = []
                
                for b_idx, batch in enumerate(coarse_batches):
                    if progress_cb:
                        completed_pct = 20 + int((b_idx / len(coarse_batches)) * 30)
                        progress_cb(completed_pct, f"Screenshot coarse {coarse_model_name}: batch {b_idx + 1}/{len(coarse_batches)} ({len(batch)} frames)")
                    
                    try:
                        detections = self.inspect_batch(
                            batch, api_key, coarse_model_name, is_coarse=True,
                            job_dir=job_dir, trace_name=f"screenshot_coarse_batch_{b_idx + 1:03d}", trace_stage="screenshot_coarse"
                        )
                        kept_in_batch = 0
                        for det in detections:
                            if not keep_coarse_detection(det, extreme_recall=extreme_recall, hybrid=False, is_two_pass=is_two_pass, exclude_male_subject=exclude_male, drop_weak_coarse=drop_weak_coarse):
                                continue

                            frame_id = det.get("frame_id")
                            matched_frame = next((f for f in batch if f["id"] == frame_id), None)
                            if matched_frame:
                                dest_name = f"{matched_frame['id']}_{matched_frame['timestamp_str'].replace(':', '_')}.jpg"
                                dest_path = job_dir / dest_name
                                shutil.copy2(matched_frame["file_path"], dest_path)
                                
                                det["image_file"] = dest_name
                                det["seconds"] = matched_frame["seconds"]
                                det["timestamp"] = matched_frame["timestamp_str"]
                                coarse_detections.append(det)
                                kept_in_batch += 1
                        if progress_cb:
                            progress_cb(
                                completed_pct,
                                f"Screenshot coarse {coarse_model_name}: batch {b_idx + 1}/{len(coarse_batches)} returned {len(detections)} raw, kept {kept_in_batch}; total candidates {len(coarse_detections)}"
                            )
                    except Exception as e:
                        raise RuntimeError(f"Error inspecting coarse frame batch {b_idx + 1}: {e}") from e
                        
                fine_detections = []
                if is_two_pass and (coarse_detections or subtitle_matches):
                    if progress_cb:
                        progress_cb(55, "Formulating fine zoom windows...")
                        
                    intervals = []
                    for det in coarse_detections:
                        start, end = self.fine_window_for_detection(det, coarse_interval)
                        intervals.append((start, end))
                        
                    for match in subtitle_matches:
                        t = match.get("seconds")
                        if t is not None:
                            start = max(0.0, t - (coarse_interval / 2.0))
                            end = t + (coarse_interval / 2.0)
                            intervals.append((start, end))
                            
                    merged_intervals = self.merge_intervals(intervals)
                    
                    if progress_cb:
                        progress_cb(60, f"Extracting fine frames at {fine_interval}s interval within {len(merged_intervals)} zoom window(s)...")
                        
                    fine_frames = []
                    for idx, (start, end) in enumerate(merged_intervals):
                        try:
                            extracted = self.extract_frames_range(
                                actual_video_path,
                                temp_dir,
                                start,
                                end,
                                fine_interval,
                                prefix=f"frame_fine_{idx}"
                            )
                            fine_frames.extend(extracted)
                        except Exception as e:
                            print(f"Error extracting fine frames for range {start}-{end}: {e}")
                            
                    total_fine = len(fine_frames)
                    if total_fine > 0:
                        if progress_cb:
                            progress_cb(70, f"Extracted {total_fine} fine frames. Analyzing...")
                            
                        fine_batches = [fine_frames[i:i + fine_batch_size] for i in range(0, total_fine, fine_batch_size)]
                        
                        for b_idx, batch in enumerate(fine_batches):
                            if progress_cb:
                                completed_pct = 70 + int((b_idx / len(fine_batches)) * 25)
                                progress_cb(completed_pct, f"Analyzing fine frames: batch {b_idx + 1}/{len(fine_batches)}...")
                                
                            try:
                                detections = self.inspect_batch(
                                    batch, api_key, fine_model_name,
                                    job_dir=job_dir, trace_name=f"screenshot_fine_batch_{b_idx + 1:03d}", trace_stage="screenshot_fine"
                                )
                                kept_in_batch = 0
                                keep_low = is_truthy_setting(get_setting("INSPECTOR_KEEP_LOW_FINE", "1"))
                                for det in detections:
                                    if not keep_fine_detection(det, extreme_recall=extreme_recall, require_concrete_evidence=require_concrete_evidence, keep_low_fine=keep_low, exclude_male_subject=exclude_male):
                                        continue

                                    frame_id = det.get("frame_id")
                                    matched_frame = next((f for f in batch if f["id"] == frame_id), None)
                                    if matched_frame:
                                        dest_name = f"{matched_frame['id']}_{matched_frame['timestamp_str'].replace(':', '_')}.jpg"
                                        dest_path = job_dir / dest_name
                                        shutil.copy2(matched_frame["file_path"], dest_path)
                                        
                                        det["image_file"] = dest_name
                                        det["seconds"] = matched_frame["seconds"]
                                        det["timestamp"] = matched_frame["timestamp_str"]
                                        fine_detections.append(det)
                                        kept_in_batch += 1
                                if progress_cb:
                                    progress_cb(
                                        completed_pct,
                                        f"Screenshot fine {fine_model_name}: batch {b_idx + 1}/{len(fine_batches)} returned {len(detections)} raw, kept {kept_in_batch}; total detections {len(fine_detections)}"
                                    )
                            except Exception as e:
                                raise RuntimeError(f"Error inspecting fine frame batch {b_idx + 1}: {e}") from e
                    if not fine_detections and coarse_detections and self.keep_coarse_when_fine_empty():
                        fine_detections = self.coarse_fallback_detections(coarse_detections)
                        if progress_cb:
                            progress_cb(
                                90,
                                f"Fine pass kept 0 detections; preserving {len(fine_detections)} coarse candidate(s) for review"
                            )
                                
            if is_two_pass:
                visual_detections = fine_detections
            else:
                visual_detections = coarse_detections

        if verify_enabled and visual_detections:
            try:
                if progress_cb:
                    progress_cb(95, f"Verifier policy={get_setting('INSPECTOR_VERIFY_POLICY', 'medium')}: {verify_model_name} reviewing uncertain detections among {len(visual_detections)} candidate(s)")
                visual_detections = self.verify_visual_detections(
                    visual_detections, job_dir, api_key, verify_model_name, progress_cb
                )
            except Exception as e:
                raise RuntimeError(f"Error verifying visual detections: {e}") from e

        # Sort and deduplicate all visual detections by confidence descending, then by timestamp (seconds)
        # Temporal corroboration (opt-in, route 3): suppress transient
        # single-frame candidates by requiring several coarse candidates near a
        # detection's time. Default min_support=1 -> no-op. High-confidence hits
        # are protected so a lone clear close-up is never dropped.
        min_scene_support = get_int_setting("INSPECTOR_MIN_SCENE_SUPPORT", 1, min_value=1, max_value=20)
        if min_scene_support > 1 and visual_detections:
            support_window = get_float_setting("INSPECTOR_SCENE_SUPPORT_WINDOW_SECONDS", float(coarse_interval) * 2.0, min_value=1.0, max_value=120.0)
            annotate_scene_support(visual_detections, coarse_detections, support_window)
            before = len(visual_detections)
            visual_detections = filter_by_scene_support(visual_detections, min_support=min_scene_support)
            if progress_cb and len(visual_detections) != before:
                progress_cb(96, f"Temporal corroboration: kept {len(visual_detections)}/{before} (min_support={min_scene_support})")

        dedup_window = get_setting("INSPECTOR_DEDUP_WINDOW", 3.0)
        deduped_detections = finalize_detections(visual_detections, dedup_window=dedup_window)

        # Semantic reason filter: a small local LLM reads each candidate's reason
        # and drops the ones it judges as clearly "no stockings" (covered legs,
        # sandals, bare feet, non-people) — robust where keyword matching fails.
        reason_filter_stats = None
        if is_truthy_setting(get_setting("INSPECTOR_REASON_FILTER", "0")) and deduped_detections:
            try:
                from .reason_filter import filter_detections_by_reason
                from .cache import Cache, default_cache_path
                _rf_cache = Cache(default_cache_path())
                deduped_detections, reason_filter_stats = filter_detections_by_reason(
                    deduped_detections, cache=_rf_cache, progress_cb=progress_cb)
                _rf_cache.close()
            except Exception as e:
                print(f"Reason filter skipped: {e}")

        # Crop-zoom enrichment: pose-localize the leg/foot, crop it from the
        # native-res frame, save as higher-res review evidence + an optional
        # re-score. Opt-in (INSPECTOR_CROP_ZOOM) and fail-open — it never changes
        # keep/drop unless INSPECTOR_CROP_ZOOM_DROP_BELOW is set. The crop is the
        # proven lever for the smooth-bare-leg-vs-肉色丝袜 class (resolution, not
        # motion); see crop_zoom.py.
        if is_truthy_setting(get_setting("INSPECTOR_CROP_ZOOM", "0")) and deduped_detections:
            try:
                deduped_detections = self._apply_crop_zoom(
                    deduped_detections, actual_video_path, job_dir, api_key,
                    coarse_model_name, progress_cb)
            except Exception as e:
                print(f"Crop-zoom skipped: {e}")

        inspection_trace = []
        trace_path = job_dir / "trace.json"
        if trace_path.exists():
            try:
                inspection_trace = json.loads(trace_path.read_text(encoding="utf-8"))
                if not isinstance(inspection_trace, list):
                    inspection_trace = []
            except Exception:
                inspection_trace = []
        if progress_cb:
            progress_cb(
                99,
                f"Final result: {len(visual_detections)} verified candidate(s), {len(deduped_detections)} after {dedup_window}s scene dedupe"
            )

        # Save the final results to job_dir/result.json
        result = {
            "video_file": host_video_path.name,
            "container_path": container_path,
            "interval": interval,
            "api_provider": self._api_provider(),
            "model": coarse_model_name,
            "fine_model": fine_model_name,
            "verify_model": verify_model_name if verify_enabled else "",
            "subtitle_matches": subtitle_matches,
            "visual_detections": deduped_detections,
            "detections_count": len(deduped_detections),
            "matches_count": len(subtitle_matches),
            "local_prefilter": local_prefilter_stats,
            "reason_filter": reason_filter_stats,
            "inspection_trace": inspection_trace,
            "manual_marks": list_manual_marks(container_path),
        }
        
        with open(job_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if progress_cb:
            progress_cb(100, "Completed")
            
        return result

    def _apply_crop_zoom(self, detections, video_path, job_dir, api_key, model_name, progress_cb=None):
        """For each kept detection, pose-localize the leg/foot and crop it from
        the native-res frame at the detection's timestamp. Attaches:
          - ``crop_image_file``: zoomed crop saved in job_dir (review evidence)
          - ``crop_pose``: whether the detector localized legs (vs heuristic crop)
          - ``crop_score``: optional VLM re-score on the crop (0-1)
        Default: enrich only. Drops a detection only if INSPECTOR_CROP_ZOOM_DROP_BELOW
        is set and the re-score falls below it.

        Pose detection (which loads opencv) runs in an isolated subprocess so it
        never clashes with PyAV in this process; if that subprocess is
        unavailable we fall back to an opencv-free heuristic crop here."""
        from . import crop_zoom
        job_dir = Path(job_dir)
        rescore = is_truthy_setting(get_setting("INSPECTOR_CROP_ZOOM_RESCORE", "1"))
        drop_below_raw = str(get_setting("INSPECTOR_CROP_ZOOM_DROP_BELOW", "") or "").strip()
        drop_below = None
        if drop_below_raw:
            try:
                drop_below = float(drop_below_raw)
            except ValueError:
                drop_below = None
        max_w = get_int_setting("INSPECTOR_CROP_ZOOM_MAX_WIDTH", 1600, min_value=640, max_value=3840)

        # Plan a crop for every detection that has a timestamp.
        plan = []  # (det, seconds, crop_name)
        for i, det in enumerate(detections, 1):
            seconds = det.get("seconds")
            if seconds is None:
                continue
            crop_name = f"{det.get('frame_id') or det.get('image_file') or 'det'}_{i:03d}_crop.jpg"
            plan.append((det, float(seconds), crop_name.replace("/", "_").replace(":", "_")))

        # Pose crops in one isolated subprocess (keeps opencv out of this process).
        items = [{"seconds": s, "out_path": str(job_dir / n)} for (_, s, n) in plan]
        results = crop_zoom.make_crops_via_subprocess(video_path, items, max_w=max_w)
        if results is None or len(results) != len(plan):
            results = [None] * len(plan)  # worker unavailable -> all heuristic fallback

        total = len(plan)
        for idx, (det, seconds, crop_name) in enumerate(plan):
            info = results[idx]
            if info is None:  # subprocess skipped this one (or failed) -> opencv-free crop
                try:
                    info = crop_zoom.make_crop_heuristic(video_path, seconds, job_dir / crop_name, max_w=max_w)
                except Exception as e:
                    print(f"Crop-zoom failed for {seconds}s: {e}")
                    info = None
            if info is None:
                continue
            det["crop_image_file"] = crop_name
            det["crop_pose"] = info["used_pose"]
            if info.get("box"):
                det["crop_box"] = info["box"]  # normalized (x0,y0,x1,y1) for UI overlay

            if rescore:
                try:
                    batch = [{"id": "crop", "file_path": str(job_dir / crop_name),
                              "timestamp_str": det.get("timestamp") or "00:00:00"}]
                    out = self.inspect_batch(batch, api_key, model_name, is_coarse=True,
                                             prompt_override=crop_zoom.CROP_RESCORE_PROMPT)
                    for d in (out or []):
                        if isinstance(d, dict) and d.get("score") is not None:
                            try:
                                det["crop_score"] = max(0.0, min(1.0, float(d.get("score"))))
                            except (TypeError, ValueError):
                                pass
                            if d.get("reason"):
                                det["crop_reason"] = d.get("reason")
                            break
                except Exception as e:
                    print(f"Crop-zoom re-score failed for {seconds}s: {e}")
            if progress_cb:
                progress_cb(99, f"Crop-zoom: {idx + 1}/{total} candidates localized")

        if drop_below is not None:
            kept = [d for d in detections
                    if not (d.get("crop_score") is not None and d["crop_score"] < drop_below)]
            if len(kept) != len(detections):
                print(f"Crop-zoom gate (<{drop_below}): kept {len(kept)}/{len(detections)}")
            return kept
        return detections

class VideoInspectorJobManager:
    def __init__(self):
        self.inspector = VideoInspector()
        self.jobs = {}
        self.lock = threading.Lock()
        self._load_persisted_jobs()

    def _load_persisted_jobs(self):
        inspections_dir = get_local_video_dir() / "inspections"
        if not inspections_dir.exists():
            return
            
        for job_dir in inspections_dir.iterdir():
            if job_dir.is_dir():
                result_json = job_dir / "result.json"
                job_status_json = job_dir / "job_status.json"
                
                if job_status_json.exists():
                    try:
                        status_data = json.loads(job_status_json.read_text(encoding="utf-8"))
                        # Ensure we don't load a running job as running if the server crashed
                        needs_save = False
                        if status_data.get("status") in ("queued", "running"):
                            status_data["status"] = "failed"
                            status_data["error"] = "Server was restarted during execution."
                            needs_save = True
                        if status_data.get("status") == "failed" and status_data.get("error") and not status_data.get("error_info"):
                            status_data["error_info"] = explain_inspector_error(status_data.get("error"))
                            needs_save = True
                        
                        # Load results if completed
                        if status_data.get("status") == "completed" and result_json.exists():
                            status_data["result"] = json.loads(result_json.read_text(encoding="utf-8"))
                            
                        self.jobs[status_data["job_id"]] = status_data
                        if needs_save:
                            self._save_job_status(status_data["job_id"])
                    except Exception as e:
                        print(f"Failed to load persisted job {job_dir.name}: {e}")

    def start_job(self, container_path, interval):
        job_id = str(uuid.uuid4())
        job_info = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "progress_tail": [format_progress_line(0, "Queued")],
            "created_at": int(time.time()),
            "container_path": container_path,
            "interval": interval,
            "result": None,
            "error": None,
        }
        
        with self.lock:
            # Clean up old jobs for the same container_path to avoid orphaned
            # files — but RETAIN the trace.json/frames of any video that the
            # regression corpus has annotated, since the offline replay harness
            # (inspector_replay Tier A) reconstructs detections from those traces.
            protected = _corpus_protected_container_paths()
            existing_jobs = [jid for jid, j in self.jobs.items() if j.get("container_path") == container_path]
            for jid in existing_jobs:
                if container_path in protected:
                    self.jobs.pop(jid, None)  # drop from the in-memory list, keep files on disk
                    print(f"Retaining job {jid} files for {container_path}: referenced by the regression corpus")
                    continue
                try:
                    self.jobs.pop(jid, None)
                    job_dir = get_local_video_dir() / "inspections" / jid
                    if job_dir.exists():
                        import shutil
                        shutil.rmtree(job_dir)
                except Exception as e:
                    print(f"Error cleaning up old job {jid} for {container_path}: {e}")
                    
            self.jobs[job_id] = job_info
            self._save_job_status(job_id)

        thread = threading.Thread(target=self._run_job, args=(job_id, container_path, interval), daemon=True)
        thread.start()
        return job_id

    def _run_job(self, job_id, container_path, interval):
        self._update_job(job_id, status="running", progress=1, message="Starting")

        def progress_cb(progress, message):
            self._update_job(job_id, status="running", progress=progress, message=message)

        try:
            result = self.inspector.run_inspection(container_path, interval, job_id, progress_cb=progress_cb)
            self._update_job(
                job_id,
                status="completed",
                progress=100,
                message="Completed",
                result=result,
            )
        except Exception as exc:
            error_text = str(exc)
            self._update_job(
                job_id,
                status="failed",
                progress=100,
                message="Failed",
                error=error_text,
                error_info=explain_inspector_error(error_text),
            )

    def _update_job(self, job_id, **fields):
        with self.lock:
            if job_id in self.jobs:
                message = fields.get("message")
                progress = fields.get("progress", self.jobs[job_id].get("progress", 0))
                if message:
                    tail = list(self.jobs[job_id].get("progress_tail") or [])
                    line = format_progress_line(progress, message)
                    if not tail or tail[-1] != line:
                        tail.append(line)
                    self.jobs[job_id]["progress_tail"] = tail[-120:]
                self.jobs[job_id].update(fields)
                self._save_job_status(job_id)

    def _save_job_status(self, job_id):
        job_dir = get_local_video_dir() / "inspections" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Don't serialize the full nested result payload in the status file to keep it clean
        status_to_save = self.jobs[job_id].copy()
        if status_to_save.get("result"):
            status_to_save["result"] = {
                "video_file": status_to_save["result"].get("video_file"),
                "detections_count": len(status_to_save["result"].get("visual_detections", [])),
                "matches_count": len(status_to_save["result"].get("subtitle_matches", []))
            }
            
        with open(job_dir / "job_status.json", "w", encoding="utf-8") as f:
            json.dump(status_to_save, f, ensure_ascii=False, indent=2)

    def get_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
                
            # If the job is completed and we only saved the summary in the memory status, reload the full result
            if job.get("status") == "completed" and (job.get("result") is None or "visual_detections" not in job["result"]):
                result_json = get_local_video_dir() / "inspections" / job_id / "result.json"
                if result_json.exists():
                    try:
                        job["result"] = json.loads(result_json.read_text(encoding="utf-8"))
                    except Exception as e:
                        print(f"Error loading full result for job {job_id}: {e}")
            if job.get("result") and job["result"].get("container_path"):
                job["result"]["manual_marks"] = list_manual_marks(job["result"].get("container_path"))
            return job

    def list_jobs(self):
        with self.lock:
            return sorted(list(self.jobs.values()), key=lambda x: x.get("created_at", 0), reverse=True)
