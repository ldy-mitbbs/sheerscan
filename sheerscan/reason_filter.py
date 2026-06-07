"""Semantic reason filter for the stockings inspector.

The vision model writes a free-text Chinese ``reason`` for every candidate. Rather
than brittle keyword matching on that prose, we ask a small **local** LLM (Ollama,
e.g. ``qwen2.5:3b``) a single question: *does this description ultimately say the
frame has (or plausibly has) thin nude/grey hosiery, or not?* — robust to phrasing
("不符合", "更像光脚穿凉鞋", "无丝袜特征", "广告图/家具" …) in a way keywords can't be.

Verdicts: ``yes`` (has / plausibly has) → keep, ``no`` (clearly excluded) → drop,
``uncertain`` (too far/dark to tell) → keep (for human review). Runs one reason
per call (small models can't reliably emit a long JSON array) and caches by reason
text so re-runs are free.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from .runtime import get_setting
from .ollama import Ollama, OllamaError

_SYSTEM = (
    "你是中文文本判别器。给你一段对视频画面的描述，判断它的【最终结论】是说画面里有"
    "（或疑似有）薄款肉色/肤色或灰色丝袜/裤袜，还是没有。描述可能先提到腿脚、最后才否定"
    "（如：腿被长裤完全覆盖、明显穿凉鞋裸脚、清楚裸足/脚趾甲可见、广告图或家具等非人物、"
    "无丝袜特征）——一律以最终结论为准。"
)


def _build_prompt(reason: str) -> str:
    return (
        "判断下面这段描述：\n"
        "- yes = 有或疑似有薄款肉色/灰色丝袜；\n"
        "- no = 明确没有或被排除（腿被长裤完全覆盖、明显穿凉鞋裸脚、清楚裸足或脚趾甲可见、"
        "非人物如广告/家具、无丝袜特征）；\n"
        "- uncertain = 太远、太暗或太模糊无法判断。\n"
        '只返回 JSON：{"r":"yes|no|uncertain"}\n\n描述：' + str(reason or "")
    )


def reason_filter_host() -> str:
    gpu = get_setting("GPU_BASE_URL", None)
    if gpu:
        return f"{str(gpu).rstrip('/').rstrip(':')}:11434"
    return "http://localhost:11434"


def reason_filter_model() -> str:
    return get_setting("INSPECTOR_REASON_FILTER_MODEL", "qwen2.5:3b")


def _normalize(value) -> str:
    v = str(value or "").strip().lower()
    if v.startswith("y"):
        return "yes"
    if v.startswith("n"):
        return "no"
    return "uncertain"


class ReasonClassifier:
    """Caches an Ollama client + per-reason verdicts for one inspection run."""

    def __init__(self, cache=None):
        self._ollama = Ollama(model=reason_filter_model(), host=reason_filter_host(), cache=cache, timeout=30.0)
        self._memo: dict[str, str] = {}
        self.available = self._ollama.ping()

    def classify(self, reason: str) -> str:
        text = str(reason or "").strip()
        if not text:
            return "uncertain"
        if text in self._memo:
            return self._memo[text]
        cache_key = "reasonfilter:" + hashlib.sha1(text.encode("utf-8")).hexdigest()
        try:
            out = self._ollama.generate_json(_build_prompt(text), system=_SYSTEM, cache_key=cache_key)
            verdict = _normalize(out.get("r") or out.get("verdict") or out.get("v"))
        except (OllamaError, AttributeError, Exception):
            # On any failure, do NOT drop the candidate — fail open to human review.
            verdict = "uncertain"
        self._memo[text] = verdict
        return verdict


def filter_detections_by_reason(detections: list[dict], *, cache=None, progress_cb=None) -> tuple[list[dict], dict]:
    """Drop detections whose reason a local LLM judges as clearly "no".

    Annotates every detection with ``reason_verdict``. Keeps ``yes`` and
    ``uncertain``. Returns ``(kept, stats)``. If the local model is unreachable,
    returns the input unchanged (fail open) so a missing Ollama never silently
    discards candidates.
    """
    clf = ReasonClassifier(cache=cache)
    if not clf.available:
        return detections, {"available": False, "input": len(detections), "kept": len(detections), "dropped": 0}
    kept: list[dict] = []
    dropped = 0
    for i, det in enumerate(detections):
        verdict = clf.classify(det.get("reason", ""))
        det["reason_verdict"] = verdict
        if verdict == "no":
            dropped += 1
        else:
            kept.append(det)
        if progress_cb and (i % 10 == 0 or i == len(detections) - 1):
            progress_cb(97, f"Reason filter ({reason_filter_model()}): {i + 1}/{len(detections)} judged, {dropped} dropped")
    return kept, {"available": True, "model": reason_filter_model(), "input": len(detections), "kept": len(kept), "dropped": dropped}
