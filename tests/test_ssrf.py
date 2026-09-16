"""test_ssrf.py — fetch_webpage 的 SSRF 防护纯函数测试（不发起真实抓取）。

is_blocked_url 对 IP 字面量与 localhost/内网域名无需外网即可判定；
公网域名单独用 example.com 验证放行（若离线无法解析会被当作阻断，
故该用例仅在能联网时断言 False，避免 CI 网络抖动导致误红）。
"""
import pytest

from tools import is_blocked_url


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",   # 云元数据
    "http://127.0.0.1:8501/admin",                # 回环
    "http://localhost/x",                         # localhost
    "http://0.0.0.0/anything",                    # unspecified
    "http://10.0.0.5/internal",                   # 私网 A
    "http://172.16.31.9/private",                 # 私网 B
    "http://192.168.1.1/router",                  # 私网 C
    "http://[::1]/ipv6loop",                      # IPv6 回环
    "file:///etc/passwd",                         # 非 http(s)
    "gopher://127.0.0.1:11211/x",                 # 危险协议
    "http://metadata.google.internal/x",          # GCP 元数据域名
    "http://db.internal/corp",                    # *.internal
])
def test_blocks_internal_targets(url):
    assert is_blocked_url(url) is True


def test_allows_public_arxiv():
    # arXiv 是研究流程的核心外部信源，必须放行（需 DNS）
    try:
        import socket
        socket.getaddrinfo("arxiv.org", None)
    except OSError:
        pytest.skip("无网络/DNS")
    assert is_blocked_url("https://arxiv.org/abs/2307.01419") is False


def test_malformed_is_blocked():
    assert is_blocked_url("not a url") is True
    assert is_blocked_url("") is True
