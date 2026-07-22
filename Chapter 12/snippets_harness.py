"""snippets_harness.py: Runnable illustrations for the harness section's reliability prose.

Two mechanisms the chapter describes in prose but did not show as code:

1. Retries with exponential backoff and a per-call timeout -- applied ONLY to idempotent
   (read-only) tools, with action tools gated behind an idempotency key instead.
2. The graceful-degradation ladder of Figure 12.3 -- primary vendor -> fallback vendor ->
   cache -> honest degradation -- fronted by a circuit breaker.

Everything here is deliberately offline and deterministic: the "vendors" are small fakes
whose failures are scripted, so the file runs in under a second with no API key and no
network, and the printed output is the same every time. The mechanics are exactly what you
would wrap around a real yfinance or market-data call.

Run it:  .venv/Scripts/python.exe snippets_harness.py
"""

from __future__ import annotations

import asyncio
import time


# --- 1. Retries, timeouts, and idempotency ------------------------------------------------
async def call_tool_reliably(tool, *, timeout_s: float = 5.0,
                             max_attempts: int = 3, base_delay_s: float = 0.1):
    """Run one IDEMPOTENT tool call under the harness's reliability policy."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            # asyncio.to_thread keeps a blocking vendor call from freezing the event loop;
            # wait_for turns a hang into a TimeoutError the retry policy can handle.
            return await asyncio.wait_for(asyncio.to_thread(tool), timeout=timeout_s)
        except Exception as exc:                       # timeout, 429, connection error...
            last_error = exc
            if attempt < max_attempts:
                delay = base_delay_s * (2 ** (attempt - 1))   # 0.1s, 0.2s, 0.4s, ...
                await asyncio.sleep(delay)
    raise RuntimeError(f"tool failed after {max_attempts} attempts") from last_error


_SUBMITTED_ORDERS: dict[str, dict] = {}    # stands in for the downstream system's dedup store


def submit_action(order: dict, idempotency_key: str) -> dict:
    """Submit a NON-idempotent action safely: the idempotency key lets the downstream
    system recognise a duplicate and return the original result instead of acting twice."""
    if idempotency_key in _SUBMITTED_ORDERS:
        return {**_SUBMITTED_ORDERS[idempotency_key], "duplicate": True}
    receipt = {"order_id": f"ORD-{len(_SUBMITTED_ORDERS) + 1:04d}", **order, "duplicate": False}
    _SUBMITTED_ORDERS[idempotency_key] = receipt
    return receipt


# --- 2. Circuit breaker + graceful-degradation ladder (Figure 12.3) -----------------------
class CircuitBreaker:
    """Stop hammering a dependency that is clearly down.

    CLOSED: calls pass through; consecutive failures are counted.
    OPEN:   after `failure_threshold` consecutive failures, calls fail fast (no timeout
            paid) for `cooloff_s` seconds.
    HALF-OPEN: after the cool-off, one trial call is allowed; success closes the breaker."""

    def __init__(self, failure_threshold: int = 3, cooloff_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooloff_s = cooloff_s
        self.consecutive_failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "CLOSED"
        if time.monotonic() - self.opened_at >= self.cooloff_s:
            return "HALF-OPEN"
        return "OPEN"

    def call(self, fn):
        if self.state == "OPEN":
            raise ConnectionError("circuit OPEN: failing fast, not waiting on a dead vendor")
        try:
            result = fn()
        except Exception:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.opened_at = time.monotonic()
            raise
        self.consecutive_failures = 0
        self.opened_at = None
        return result


def get_ratios_with_degradation(ticker: str, primary, fallback,
                                cache: dict, breaker: CircuitBreaker) -> dict:
    """Fetch ratios down the degradation ladder. Whatever rung answers, the result SAYS
    which rung it came from -- degradation must be visible, never silent."""
    try:
        ratios = breaker.call(lambda: primary(ticker))
        cache[ticker] = ratios                                   # refresh the last-good value
        return {"ratios": ratios, "source": "primary", "caveat": None}
    except Exception:
        pass                                                     # fall through the ladder
    try:
        ratios = fallback(ticker)
        return {"ratios": ratios, "source": "fallback", "caveat": "secondary source"}
    except Exception:
        pass
    if ticker in cache:
        return {"ratios": cache[ticker], "source": "cache",
                "caveat": "stale last-known value -- discount accordingly"}
    return {"ratios": None, "source": "degraded",
            "caveat": "INSUFFICIENT_EVIDENCE: no ratio source available; escalate to a human"}


# --- Demo ----------------------------------------------------------------------------------
def main() -> None:
    print("1) retry with backoff on an idempotent tool (vendor fails twice, then recovers)")
    calls = {"n": 0}

    def flaky_ratios_fetch() -> dict:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("HTTP 429 rate-limited")
        return {"ticker": "MSFT", "trailing_pe": 35.1}

    result = asyncio.run(call_tool_reliably(flaky_ratios_fetch, timeout_s=2.0))
    print(f"    recovered on attempt {calls['n']}: {result}\n")

    print("2) an action tool is NOT blindly retried -- the idempotency key absorbs the retry")
    order = {"ticker": "MSFT", "side": "BUY", "qty": 100}
    first = submit_action(order, idempotency_key="run42-MSFT-hedge")
    retry = submit_action(order, idempotency_key="run42-MSFT-hedge")   # same key: no 2nd ticket
    print(f"    first submit : {first}")
    print(f"    retried call : {retry}\n")

    print("3) circuit breaker + degradation ladder during a primary-vendor outage")
    breaker = CircuitBreaker(failure_threshold=3, cooloff_s=30.0)
    cache = {"MSFT": {"ticker": "MSFT", "trailing_pe": 34.8, "as_of": "yesterday"}}

    def dead_primary(ticker: str) -> dict:
        raise TimeoutError("primary vendor timeout")

    def flaky_fallback(ticker: str) -> dict:
        raise ConnectionError("fallback vendor 503")

    for run in range(1, 5):
        print(f"  run {run} (breaker {breaker.state}):")
        out = get_ratios_with_degradation("MSFT", dead_primary, flaky_fallback, cache, breaker)
        print(f"    -> source={out['source']}  caveat={out['caveat']}")
    print(f"  breaker is now {breaker.state}: runs 4+ skip the primary's timeout entirely")

if __name__ == "__main__":
    main()
