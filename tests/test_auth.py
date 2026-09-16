"""test_auth.py — web/auth.py 凭据与限流的离线测试（不依赖 fastapi）。"""
import importlib
import os

from web.auth import TokenBucket, verify_credentials, auth_configured


def _reload_auth():
    import web.auth as a
    return importlib.reload(a)


def test_verify_credentials_matches_env(monkeypatch):
    monkeypatch.setenv("WEB_USERNAME", "research")
    monkeypatch.setenv("WEB_PASSWORD", "s3cret")
    a = _reload_auth()
    assert a.auth_configured() is True
    assert a.verify_credentials("research", "s3cret") is True
    assert a.verify_credentials("research", "wrong") is False
    assert a.verify_credentials("admin", "s3cret") is False


def test_auth_not_configured_when_empty(monkeypatch):
    monkeypatch.delenv("WEB_USERNAME", raising=False)
    monkeypatch.delenv("WEB_PASSWORD", raising=False)
    a = _reload_auth()
    assert a.auth_configured() is False
    assert a.verify_credentials("", "") is False


def test_token_bucket_refills_and_caps_burst():
    # 容量 2：连发两次放行，第三次拒绝
    b = TokenBucket(rate_per_min=6000, burst=2)
    assert [b.allow("x") for _ in range(3)] == [True, True, False]
    # 不同 key 互不影响
    assert b.allow("y") is True


def test_token_bucket_refills_over_time():
    import time
    b = TokenBucket(rate_per_min=60000, burst=1)   # 每秒补 1000 个 → 极快回血
    assert b.allow("z") is True
    assert b.allow("z") is False
    time.sleep(0.05)                                # 补充若干令牌
    assert b.allow("z") is True
