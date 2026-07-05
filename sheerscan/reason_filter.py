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
import re
from typing import Optional

from .runtime import get_setting
from .ollama import Ollama, OllamaError
from .openai_chat import OpenAIChat, OpenAIChatError

_SYSTEM = (
    "你是中文文本判别器。给你一段对视频画面的描述，判断它的【最终结论】是说画面里有"
    "（或疑似有）薄款肉色/肤色或灰色丝袜/裤袜，还是没有。描述可能先提到腿脚、最后才否定"
    "（如：腿被长裤完全覆盖、明显穿凉鞋裸脚、清楚裸足/脚趾甲可见、脚趾缝清楚、"
    "广告图或家具等非人物、无丝袜特征）——一律以最终结论为准。"
)


# The coarse prompt deliberately primes the VLM toward high recall, so even
# clearly-bare feet come back with a trailing speculation clause ("无法排除肉色
# 丝袜的可能性…保留该帧供后续判断"). A 3b judge reliably flips to keep on that
# clause, defeating its own "清楚裸足→no" rule. Stripping the stereotyped hedge
# sentences BEFORE judging is deterministic and can only act when the remaining
# observation is clearly negative — a hedge-only reason strips to empty, which
# classifies as uncertain (kept for human review), so recall is preserved.
_HEDGE_SENTENCE = re.compile(
    r"(无法[^。；;!？?]{0,8}排除|不排除|不能排除"
    r"|可能(隐藏|穿着?|是|为)[^。；;!？?]{0,20}(丝袜|裤袜)"
    r"|保留[^。；;!？?]{0,10}(判断|复核|审核)"
    r"|[供需待][^。；;!？?]{0,4}后续[^。；;!？?]{0,6}(判断|复核|确认)"
    r"|后续(步骤)?(判断|复核|确认)"
    r"|易漏判|需注意|需人工|建议人工)"
)

# "（如脱鞋后仍显皮肤光泽…）"-style hypothetical clauses quote the CRITERIA, not
# the frame — drop them before hedge/observation checks so their 质感/光泽 words
# can't shield the hedge sentence from stripping.
_HYPOTHETICAL_CLAUSE = re.compile(r"[（(](?:如|例如|比如)[^）)]*[）)]|(?:例如|比如)[^。；;!？?，,]*")

# A hedge sentence is only safe to drop when it carries no actual observation.
# The coarse model often packs real evidence and the disclaimer into one long
# comma-run sentence ("皮肤质感光滑…看起来像极薄肉色丝袜（无法完全排除裸足）") —
# stripping those loses the evidence and was measured to kill ~15 true events.
_OBSERVATION_TERM = re.compile(
    r"(皮肤|质感|光泽|薄层|纹理|织物|材质|袜口|趾甲|反光)"
)


def strip_hedge_sentences(text: str) -> str:
    """Drop pure-speculation disclaimer sentences, keeping actual observations."""
    parts = re.split(r"(?<=[。；;!！？?])", str(text or ""))
    kept = []
    for p in parts:
        if not p.strip():
            continue
        gist = _HYPOTHETICAL_CLAUSE.sub("", p)
        if _HEDGE_SENTENCE.search(gist) and not _OBSERVATION_TERM.search(gist):
            continue
        kept.append(p)
    return "".join(kept).strip()


# Structured-output schema for the OpenAI/LM Studio backend: forces a valid
# {"r": "yes"|"no"|"uncertain"} reply (the Ollama backend ignores it, using
# its own format=json).
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"r": {"type": "string", "enum": ["yes", "no", "uncertain"]}},
    "required": ["r"],
}


def _build_prompt(reason: str) -> str:
    return (
        "判断下面这段描述：\n"
        "- yes = 有或疑似有薄款肉色/灰色丝袜；\n"
        "- no = 明确没有或被排除（腿被长裤完全覆盖、明显穿凉鞋裸脚、清楚裸足、"
        "脚趾/趾甲/脚趾缝清晰可见、非人物如广告/家具、无丝袜特征）；\n"
        "- uncertain = 太远、太暗或太模糊无法判断。\n"
        "示例：\n"
        '描述：脚趾清晰可见，皮肤质感自然，无明显丝袜覆盖。→{"r":"no"}\n'
        '描述：小腿和脚背覆盖连续均匀、略带光泽的肉色薄层，看不清趾缝。→{"r":"yes"}\n'
        '描述：女性穿凉鞋，趾甲清楚可见，为裸足。→{"r":"no"}\n'
        '描述：远景，光线昏暗，细节无法分辨。→{"r":"uncertain"}\n'
        '描述：腿部处于阴影中被遮挡，无法看清是否穿丝袜。→{"r":"uncertain"}\n'
        '只返回 JSON：{"r":"yes|no|uncertain"}\n\n描述：' + str(reason or "")
    )


def reason_filter_host() -> str:
    gpu = get_setting("GPU_BASE_URL", None)
    if gpu:
        return f"{str(gpu).rstrip('/').rstrip(':')}:11434"
    return "http://localhost:11434"


def reason_filter_backend() -> str:
    """Which LLM backend serves the reason filter: 'ollama' (default) or
    'openai' (any OpenAI-compatible server, e.g. LM Studio on a local GPU)."""
    return str(get_setting("INSPECTOR_REASON_FILTER_BACKEND", "ollama") or "ollama").strip().lower()


def reason_filter_base_url() -> str:
    """OpenAI-compatible base URL for the 'openai' backend. Explicit setting wins;
    otherwise derive from GPU_BASE_URL (LM Studio's :1234), else localhost."""
    url = get_setting("INSPECTOR_REASON_FILTER_BASE_URL", None)
    if url:
        return str(url)
    gpu = get_setting("GPU_BASE_URL", None)
    if gpu:
        return f"{str(gpu).rstrip('/').rstrip(':')}:1234/v1"
    return "http://localhost:1234/v1"


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
        if reason_filter_backend() == "openai":
            self._client = OpenAIChat(model=reason_filter_model(),
                                      base_url=reason_filter_base_url(),
                                      api_key=get_setting("INSPECTOR_REASON_FILTER_API_KEY", "lm-studio") or "lm-studio",
                                      cache=cache, timeout=30.0)
        else:
            self._client = Ollama(model=reason_filter_model(), host=reason_filter_host(), cache=cache, timeout=30.0)
        self._memo: dict[str, str] = {}
        self.available = self._client.ping()

    def classify(self, reason: str) -> str:
        raw = str(reason or "").strip()
        if not raw:
            return "uncertain"
        if raw in self._memo:
            return self._memo[raw]
        text = strip_hedge_sentences(raw)
        if not text:
            # Reason was pure speculation ("无法排除…保留复核") with no actual
            # observation — keep for human review rather than judging noise.
            self._memo[raw] = "uncertain"
            return "uncertain"
        # v2: hedge sentences are stripped before judging. The key must version
        # with the preprocessing/prompt or stale verdicts mask the change.
        cache_key = "reasonfilter:v2:" + hashlib.sha1(text.encode("utf-8")).hexdigest()
        try:
            out = self._client.generate_json(_build_prompt(text), system=_SYSTEM,
                                             cache_key=cache_key, schema=_VERDICT_SCHEMA)
            verdict = _normalize(out.get("r") or out.get("verdict") or out.get("v"))
        except (OllamaError, OpenAIChatError, AttributeError, Exception):
            # On any failure, do NOT drop the candidate — fail open to human review.
            verdict = "uncertain"
        self._memo[raw] = verdict
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
