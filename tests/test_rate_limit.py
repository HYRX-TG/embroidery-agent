"""速率限制器单元测试。"""
from embroidery_agent.api.rate_limit import SlidingWindowLimiter


def test_allows_within_limit():
    limiter = SlidingWindowLimiter(limits={"chat": 3})
    results = [limiter.check("chat", "1.2.3.4")[0] for _ in range(3)]
    assert results == [True, True, True]


def test_blocks_over_limit():
    limiter = SlidingWindowLimiter(limits={"chat": 3})
    for _ in range(3):
        limiter.check("chat", "1.2.3.4")
    allowed, retry_after = limiter.check("chat", "1.2.3.4")
    assert not allowed
    assert retry_after >= 1


def test_different_keys_isolated():
    limiter = SlidingWindowLimiter(limits={"chat": 1})
    assert limiter.check("chat", "ip-a")[0] is True
    assert limiter.check("chat", "ip-a")[0] is False
    assert limiter.check("chat", "ip-b")[0] is True


def test_different_buckets_isolated():
    limiter = SlidingWindowLimiter(limits={"chat": 1, "search": 10})
    assert limiter.check("chat", "ip-a")[0] is True
    assert limiter.check("chat", "ip-a")[0] is False
    assert limiter.check("search", "ip-a")[0] is True


def test_unknown_bucket_always_allows():
    limiter = SlidingWindowLimiter(limits={"chat": 1})
    assert limiter.check("nope", "ip-a")[0] is True
