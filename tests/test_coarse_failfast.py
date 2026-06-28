"""Coarse backend fail-fast: a down/IP-changed local GPU box must not wedge the
job. `_preflight_coarse_backend` health-checks a local OpenAI/LM Studio endpoint
up front and raises immediately when it's unreachable, instead of letting every
coarse batch grind through its full connect-timeout for hours."""
import pytest
import requests

import sheerscan.inspector as ins


def _patch_settings(monkeypatch, mapping):
    monkeypatch.setattr(ins, "get_setting",
                        lambda n, d=None: mapping.get(n, d))


def test_preflight_skips_cloud_provider(monkeypatch):
    # Cloud providers rely on per-batch retry/backoff; the health check must be a
    # no-op for them (and must not even touch the network).
    _patch_settings(monkeypatch, {"INSPECTOR_API_PROVIDER": "mulerouter"})
    def _boom(*a, **k):  # network must not be hit
        raise AssertionError("cloud provider should not be health-checked")
    monkeypatch.setattr(ins.requests, "get", _boom)
    ins.VideoInspector()._preflight_coarse_backend()  # no raise


def test_preflight_passes_when_backend_healthy(monkeypatch):
    _patch_settings(monkeypatch, {
        "INSPECTOR_API_PROVIDER": "openai",
        "INSPECTOR_OPENAI_BASE_URL": "http://gpu:1234/v1",
    })
    seen = {}
    class Resp:
        status_code = 200
        def close(self): pass
    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return Resp()
    monkeypatch.setattr(ins.requests, "get", fake_get)
    ins.VideoInspector()._preflight_coarse_backend()  # no raise
    assert seen["url"] == "http://gpu:1234/v1/models"
    assert seen["timeout"] == 5  # short, bounded — never hangs


def test_preflight_raises_fast_when_unreachable(monkeypatch):
    _patch_settings(monkeypatch, {
        "INSPECTOR_API_PROVIDER": "openai",
        "INSPECTOR_OPENAI_BASE_URL": "http://dead-host:1234/v1",
    })
    def fake_get(url, headers=None, timeout=None):
        raise requests.exceptions.ConnectionError("no route to host")
    monkeypatch.setattr(ins.requests, "get", fake_get)
    with pytest.raises(RuntimeError, match="unreachable"):
        ins.VideoInspector()._preflight_coarse_backend()


def test_preflight_raises_on_non_200(monkeypatch):
    _patch_settings(monkeypatch, {
        "INSPECTOR_API_PROVIDER": "lmstudio",
        "INSPECTOR_OPENAI_BASE_URL": "http://gpu:1234/v1",
    })
    class Resp:
        status_code = 503
        def close(self): pass
    monkeypatch.setattr(ins.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match="503"):
        ins.VideoInspector()._preflight_coarse_backend()
