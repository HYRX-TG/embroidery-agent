"""进程内滑动窗口速率限制（零依赖，单进程版）。

按 (IP + 桶名) 限流：滑动窗口内请求数超过上限返回 429。
注意：仅适用于单 worker 部署；多实例部署需换 Redis 等共享存储。
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, limits: dict[str, int] | None = None, window_seconds: float = 60.0):
        """limits: 桶名 → 窗口内最大请求数，如 {"chat": 10, "search": 30}。"""
        self.limits = limits or {}
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, bucket: str, key: str) -> tuple[bool, int]:
        """返回 (是否放行, 距离下次可请求的秒数)。放行时第二值为 0。"""
        limit = self.limits.get(bucket)
        if not limit or limit <= 0:
            return True, 0
        identity = f"{bucket}:{key}"
        now = time.monotonic()
        with self._lock:
            hits = self._hits[identity]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = int(self.window - (now - hits[0])) + 1
                return False, max(retry_after, 1)
            hits.append(now)
            # 防止 key 无限增长：空队列回收
            if not hits:
                self._hits.pop(identity, None)
            return True, 0

    def _prune(self) -> None:
        """清理长期未用的 key（定时调用，防内存膨胀）。"""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, dq in self._hits.items() if not dq or now - dq[-1] >= self.window * 2]
            for k in stale:
                self._hits.pop(k, None)
