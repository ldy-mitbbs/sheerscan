"""Minimal Ollama client (stdlib urllib only).

Vendored from the host app's LLM layer. The pipeline only needs a local model
for the semantic reason filter, so this carries just what ``reason_filter``
uses: ``ping()`` and ``generate_json()`` with optional verdict caching.

The optional ``cache`` argument is any object exposing
``get_llm(key, model)`` / ``put_llm(key, model, value)`` — see
:mod:`sheerscan.cache`. Pass ``None`` to disable caching.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_HOST = "http://localhost:11434"


def _loose_json_loads(text: str) -> Any:
    """json.loads with a forgiving fallback.

    Models occasionally emit JSON with un-escaped ASCII double-quotes inside
    string values. When strict parsing fails, walk the text once tracking
    string state and escape any `"` that clearly cannot be a JSON string
    terminator (i.e. followed by non-structural chars).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    out: list[str] = []
    in_str = False
    esc = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        # inside a string
        if esc:
            out.append(ch); esc = False; i += 1; continue
        if ch == "\\":
            out.append(ch); esc = True; i += 1; continue
        if ch == '"':
            # Decide if this quote terminates the string. If the next
            # non-whitespace char is one of `, } ] :` it's structural; otherwise
            # treat as an inner quote and escape it.
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            nxt = text[j] if j < n else ""
            if nxt in (",", "}", "]", ":", ""):
                out.append(ch); in_str = False
            else:
                out.append('\\"')
            i += 1; continue
        out.append(ch); i += 1
    return json.loads("".join(out))


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        cache=None,
        timeout: float = 120.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.cache = cache
        self.timeout = timeout

    # ---- low level ----
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.host}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama request failed: {e}") from e
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise OllamaError(f"Bad JSON from Ollama: {body[:200]}") from e

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ---- high level ----
    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        cache_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Any:
        """Run prompt; expect JSON back. Uses cache if cache_key provided."""
        if cache_key and self.cache:
            cached = self.cache.get_llm(cache_key, self.model)
            if cached is not None:
                return cached

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,  # disable qwen3-style thinking that yields empty responses
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        resp = self._post("/api/generate", payload)
        text = (resp.get("response") or "").strip()
        # Some thinking-models put JSON inside ```json ...``` fences; strip them.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        if not text:
            raise OllamaError(
                "LLM returned empty response (model may be a 'thinking' model; "
                "upgrade Ollama or pick a non-thinking model like qwen2.5 / llama3.1)"
            )
        try:
            parsed = _loose_json_loads(text)
        except json.JSONDecodeError as e:
            raise OllamaError(f"LLM returned non-JSON: {text[:300]!r}") from e

        if cache_key and self.cache:
            self.cache.put_llm(cache_key, self.model, parsed)
        return parsed
