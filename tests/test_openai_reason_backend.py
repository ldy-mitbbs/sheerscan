"""OpenAI-compatible reason-filter backend (LM Studio): JSON parsing + caching,
and that ReasonClassifier selects it when configured."""
import io
import json
from contextlib import contextmanager

import sheerscan.openai_chat as oc
import sheerscan.reason_filter as rf


def _fake_resp(body):
    class R:
        status = 200
        def read(self): return body.encode("utf-8")
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return R()


def test_openai_chat_parses_json_from_text(monkeypatch):
    # LM Studio replies with plain text (no response_format); we parse the object.
    chat_body = json.dumps({"choices": [{"message": {"content": 'sure: {"r":"no"}'}}]})
    monkeypatch.setattr(oc.urllib.request, "urlopen", lambda req, timeout=0: _fake_resp(chat_body))
    c = oc.OpenAIChat(model="qwen2.5-7b-instruct", base_url="http://x/v1")
    out = c.generate_json("p", system="s")
    assert out == {"r": "no"}


def test_openai_chat_uses_cache(monkeypatch):
    class Cache:
        def __init__(self): self.store = {}
        def get_llm(self, k, m): return self.store.get((k, m))
        def put_llm(self, k, m, v): self.store[(k, m)] = v
    cache = Cache()
    body = json.dumps({"choices": [{"message": {"content": '{"r":"yes"}'}}]})
    calls = {"n": 0}
    def fake(req, timeout=0):
        calls["n"] += 1
        return _fake_resp(body)
    monkeypatch.setattr(oc.urllib.request, "urlopen", fake)
    c = oc.OpenAIChat(model="m", base_url="http://x/v1", cache=cache)
    assert c.generate_json("p", cache_key="k") == {"r": "yes"}
    assert c.generate_json("p", cache_key="k") == {"r": "yes"}  # served from cache
    assert calls["n"] == 1


def test_reason_classifier_selects_openai_backend(monkeypatch):
    monkeypatch.setattr(rf, "get_setting", lambda n, d=None: {
        "INSPECTOR_REASON_FILTER_BACKEND": "openai",
        "INSPECTOR_REASON_FILTER_BASE_URL": "http://x/v1",
        "INSPECTOR_REASON_FILTER_MODEL": "qwen2.5-7b-instruct",
    }.get(n, d))
    # avoid a real ping
    monkeypatch.setattr(oc.OpenAIChat, "ping", lambda self: True)
    clf = rf.ReasonClassifier()
    assert isinstance(clf._client, oc.OpenAIChat)
    assert clf._client.model == "qwen2.5-7b-instruct"
