"""Minimal OpenAI-compatible chat client (stdlib urllib only).

Mirrors the surface ``reason_filter`` uses from :class:`sheerscan.ollama.Ollama`
(``ping()`` + ``generate_json()`` with optional verdict caching) so the semantic
reason filter can run on any OpenAI-compatible server — notably **LM Studio** on
a local GPU box, where the heavier judge model lives.

Note: LM Studio rejects ``response_format={"type":"json_object"}`` (only
``json_schema``/``text``), so we don't send it and instead parse the JSON object
out of the model's text reply (reusing the Ollama client's forgiving loader).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Optional

from .ollama import _loose_json_loads


class OpenAIChatError(RuntimeError):
    pass


class OpenAIChat:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        cache=None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "lm-studio"
        self.cache = cache
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        cache_key: Optional[str] = None,
        temperature: float = 0.0,
        schema: Optional[dict] = None,
    ) -> Any:
        if cache_key and self.cache:
            cached = self.cache.get_llm(cache_key, self.model)
            if cached is not None:
                return cached

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if schema is not None:
            # Structured output (LM Studio supports json_schema, not json_object):
            # constrains the reply to valid JSON matching the schema.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": schema},
            }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise OpenAIChatError(f"OpenAI chat request failed: {e}") from e
        try:
            choices = json.loads(body).get("choices") or []
            text = (choices[0]["message"]["content"] or "").strip()
        except Exception as e:
            raise OpenAIChatError(f"Bad response: {body[:200]}") from e

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        if not text:
            raise OpenAIChatError("empty response from chat model")
        try:
            parsed = _loose_json_loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                raise OpenAIChatError(f"non-JSON reply: {text[:200]!r}")
            parsed = _loose_json_loads(m.group(0))

        if cache_key and self.cache:
            self.cache.put_llm(cache_key, self.model, parsed)
        return parsed
