"""
web/auth.py — 凭据与限流（L2：无用户体系，用共享口令 + 令牌桶兜住滥用）

设计取舍：L2 不建数据库/账号系统，采用「Basic Auth 口令」做访问控制，
并用进程内令牌桶对提交频率做粗粒度限流，防止被脚本刷爆 LLM/Tavily 额度。
口令、限流参数全部来自环境变量，便于容器注入。生产建议再叠加 Nginx 层限流。
"""
from __future__ import annotations

import hmac
import os
import threading
import time


# ------------------------------------------------------------------ #
# Basic Auth 凭据
# ------------------------------------------------------------------ #
def expected_credentials() -> tuple[str, str]:
    """从环境读取服务端要求的用户名/口令。未配置时返回空串（由调用方决定是否放行）。"""
    return (os.getenv("WEB_USERNAME", ""), os.getenv("WEB_PASSWORD", ""))


def auth_configured() -> bool:
    u, p = expected_credentials()
    return bool(u and p)


def verify_credentials(username: str, password: str) -> bool:
    """定长时间比较，避免计时侧信道。"""
    eu, ep = expected_credentials()
    if not (eu and ep):
        return False
    return hmac.compare_digest(username or "", eu) and hmac.compare_digest(password or "", ep)


# ------------------------------------------------------------------ #
# 简单令牌桶限流（按客户端 IP）
# ------------------------------------------------------------------ #
class TokenBucket:
    """线程安全令牌桶：每分钟补充 rate 个令牌，容量 burst。用于限制研究请求提交频率。"""

    def __init__(self, rate_per_min: int, burst: int):
        self.rate = max(1.0, float(rate_per_min)) / 60.0   # 每秒补充
        self.capacity = float(max(1, burst))
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                return True
            self._buckets[key] = (tokens, now)
            return False


_limiter: TokenBucket | None = None


def get_limiter() -> TokenBucket:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucket(
            rate_per_min=int(os.getenv("RATE_LIMIT_PER_MIN", "6")),
            burst=int(os.getenv("RATE_LIMIT_BURST", "3")),
        )
    return _limiter
