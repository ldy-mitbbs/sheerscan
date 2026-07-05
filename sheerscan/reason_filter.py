"""Semantic reason filter for the stockings inspector.

The vision model writes a free-text Chinese ``reason`` for every candidate. A
local text LLM reads each reason and judges whether the *observations* in it
support hosiery — the coarse prompt deliberately primes the VLM toward high
recall, so even clearly-bare feet come back with a trailing speculation clause
("无法排除肉色丝袜的可能性…需后续判断"), and the judge is instructed to ignore
those hedges and rule on the observed evidence only.

This needs a judge that can actually follow that instruction. Measured on the
corpus (2026-07-05): ``qwen/qwen3-14b`` on the GPU box, plain prompt, no
preprocessing = recall 0.734 / precision 0.483 — vs the old lax judge's
0.739/0.381 (kept every hedged bare foot) and vs a 3b judge + regex hedge
stripping's 0.712/0.494 (regex was a phrasing arms race; a mutated hedge like
"需后续判断（如…符合丝袜特征）" walked right past it). Small models (≤3b)
cannot do this with any prompt we found — don't downgrade the model without
re-running ``inspect-replay --mode postprocess --compare``.

Verdicts: ``yes`` (observed evidence) → keep, ``no`` (clearly excluded) → drop,
``uncertain`` (can't tell from the observation) → keep (for human review). Runs
one reason per call and caches by (reason text, model) so re-runs are free.
Fail-open: if the backend is unreachable, everything is kept.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from .runtime import get_setting
from .ollama import Ollama, OllamaError
from .openai_chat import OpenAIChat, OpenAIChatError

_SYSTEM = (
    "你是丝袜检测流水线的文本裁判。上游视觉模型对每个候选画面写了一段中文描述，"
    "你根据描述判断画面是否可能有薄款肉色/肤色/灰色半透明丝袜。"
    "关键规则：上游模型被要求宁可多报，所以描述末尾经常挂着例行免责（"
    "如“无法排除肉色丝袜”“需后续判断”“保留供人工复核”“可能隐藏丝袜”等）——"
    "这些套话不是证据，判断时完全忽略，只依据描述中实际观察到的内容。/no_think"
)


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
        "根据描述中的实际观察判断：\n"
        "- yes = 观察到丝袜正证据：半透明织物覆盖皮肤、尼龙光泽薄层、袜口/缝线/网纹、"
        "脚趾细节被织物柔化，或描述明说看到丝袜；\n"
        "- no = 观察到明确排除证据：清楚的裸足（脚趾/趾甲/趾缝清晰、自然皮肤纹理）且无任何织物迹象、"
        "凉鞋裸脚、腿被长裤/长裙完全遮住看不到皮肤、非人物（广告/家具/图画）；\n"
        "- uncertain = 观察本身不足以判断（太远、太暗、太模糊、只看到部分皮肤无法分辨）。\n"
        "注意：“无法排除丝袜/需后续判断”之类的免责句不算观察，忽略之；"
        "但若观察里确实提到光泽薄层、织物感等正证据，即使措辞犹豫也算 yes。\n"
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
    """Caches an LLM client + per-reason verdicts for one inspection run."""

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
        text = str(reason or "").strip()
        if not text:
            return "uncertain"
        if text in self._memo:
            return self._memo[text]
        # v2 key: hashes the raw reason (the cache layer also keys on model).
        cache_key = "reasonfilter:v2:" + hashlib.sha1(text.encode("utf-8")).hexdigest()
        try:
            out = self._client.generate_json(_build_prompt(text), system=_SYSTEM,
                                             cache_key=cache_key, schema=_VERDICT_SCHEMA)
            verdict = _normalize(out.get("r") or out.get("verdict") or out.get("v"))
        except (OllamaError, OpenAIChatError, AttributeError, Exception):
            # On any failure, do NOT drop the candidate — fail open to human review.
            verdict = "uncertain"
        self._memo[text] = verdict
        return verdict


def filter_detections_by_reason(detections: list[dict], *, cache=None, progress_cb=None) -> tuple[list[dict], dict]:
    """Drop detections whose reason a local LLM judges as clearly "no".

    Annotates every detection with ``reason_verdict``. Keeps ``yes`` and
    ``uncertain``. Returns ``(kept, stats)``. If the local model is unreachable,
    returns the input unchanged (fail open) so a missing backend never silently
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
