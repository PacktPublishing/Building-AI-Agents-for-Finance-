"""snippets_cost.py : Runnable illustrations for the cost/latency/scaling section.

Four levers the chapter explains in prose and schemas but did not show as code:

1. The cost anatomy of Figure 12.9: Every iteration re-sends the growing context, so
   the bill grows quadratically-ish with loop length; prompt caching discounts the stable
   prefix that is re-sent on every trip.
2. A tool-result cache that turns repeated fetches within a batch into one vendor call.
3. A model cascade (Figure 12.10): the SLM first, escalate to gpt-4o only on low
   confidence, with the blended cost compared against running everything on gpt-4o.
4. A fan-out sized to a rate limit with an asyncio semaphore; parallelism collapses
   wall-clock time only up to the concurrency the rate limit allows.

Offline and deterministic: the model prices are the real published ones, imported from
tracing.py; the tasks and latencies are scripted. 

Run it: .venv/Scripts/python.exe snippets_cost.py
"""

from __future__ import annotations

import asyncio
import time

from tracing import PRICE_PER_MTOK, cost_usd

# --- 1. The cost anatomy of a run (Figure 12.9) -----------------------------------
def estimate_run_cost(model: str, *, system_tokens: int, question_tokens: int,
                      tool_result_tokens: int, output_tokens_per_turn: int,
                      iterations: int, cached_prefix_discount: float = 0.0) -> dict:
    """Price a multi-iteration run the way the API actually bills it.

    Every iteration re-sends the entire conversation so far, so the stable prefix (system
    prompt + tool definitions) is paid once PER ITERATION, and each tool result joins the
    context that every LATER iteration re-sends. `cached_prefix_discount` models provider
    prompt caching: 0.0 = no caching, 0.5 = the re-sent stable prefix costs half."""
    in_price, out_price = PRICE_PER_MTOK[model]
    total_input = 0
    context = system_tokens + question_tokens                 # what iteration 1 sends
    for i in range(1, iterations + 1):
        billable = context
        if i > 1 and cached_prefix_discount:                   # prefix cached after 1st send
            billable -= system_tokens * cached_prefix_discount
        total_input += billable
        context += output_tokens_per_turn + tool_result_tokens  # next turn re-sends all this
    total_output = output_tokens_per_turn * iterations
    return {"input_tokens": int(total_input), "output_tokens": total_output,
            "cost_usd": round((total_input * in_price + total_output * out_price) / 1e6, 6)}


# --- 2. Tool-result caching -------------------------------------------------------------------
class ToolResultCache:
    """Serve a repeated (tool, arguments) fetch from memory instead of the vendor.

    Safe because the underlying data is deterministic for the day; the win is fewer vendor
    calls, lower latency, and no rate-limit pressure from redundant fetches."""

    def __init__(self):
        self._store: dict = {}
        self.hits = 0
        self.misses = 0

    def fetch(self, tool_name: str, ticker: str, vendor_call):
        key = (tool_name, ticker)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1                                # only a miss pays the vendor
        self._store[key] = vendor_call(ticker)
        return self._store[key]


# --- 3. A model cascade: SLM first, escalate on low confidence (Figure 12.10)--------------------------------
def run_cascade(tasks: list[dict], *, confidence_floor: float = 0.8) -> dict:
    """Route each task through the cascade of Figure 12.11 and account for the cost."""
    blended = 0.0
    gpt4o_only = 0.0
    escalated = 0
    for task in tasks:
        tin, tout = task["input_tokens"], task["output_tokens"]
        blended += cost_usd("gpt-4o-mini", tin, tout)          # the SLM always tries first
        if task["slm_confidence"] < confidence_floor:          # uncertain -> escalate
            blended += cost_usd("gpt-4o", tin, tout)
            escalated += 1
        gpt4o_only += cost_usd("gpt-4o", tin, tout)
    return {"tasks": len(tasks), "escalated": escalated,
            "blended_cost_usd": round(blended, 4),
            "gpt4o_only_cost_usd": round(gpt4o_only, 4),
            "saving_pct": round(100 * (1 - blended / gpt4o_only), 1)}


# --- 4. Fan-out sized to a rate limit ----------------------------------------------------------
async def fan_out(tickers: list[str], analyse_one, max_concurrent: int) -> float:
    """Run the batch in parallel, but never more than `max_concurrent` in flight.

    The semaphore is the fan-out sized to the rate limit: enough parallelism to make the
    deadline, not so much that the provider throttles and the batch spends its time backing
    off from 429s. Returns wall-clock seconds."""
    gate = asyncio.Semaphore(max_concurrent)

    async def bounded(ticker: str):
        async with gate:
            return await analyse_one(ticker)

    start = time.perf_counter()
    await asyncio.gather(*(bounded(t) for t in tickers))
    return time.perf_counter() - start


# --- Demo ---------------------------------------------------------------------------------------
def main() -> None:
    print("1) cost anatomy: same run, growing loop length (gpt-4o-mini, tokens like the lab's)")
    shape = dict(system_tokens=600, question_tokens=20,
                 tool_result_tokens=250, output_tokens_per_turn=120)
    for iters in (2, 4, 6):
        plain = estimate_run_cost("gpt-4o-mini", iterations=iters, **shape)
        cached = estimate_run_cost("gpt-4o-mini", iterations=iters,
                                   cached_prefix_discount=0.5, **shape)
        print(f"    {iters} iterations: {plain['input_tokens']:>5} input tokens, "
              f"${plain['cost_usd']:.6f}  (with prompt caching: ${cached['cost_usd']:.6f})")
    print("    -> input tokens grow much faster than the answer does; the loop, not the")
    print("       answer, is what you are paying for\n")

    print("2) tool-result cache across a 5-name batch that re-checks MSFT twice")
    cache = ToolResultCache()
    vendor_calls = {"n": 0}

    def vendor(ticker: str) -> dict:
        vendor_calls["n"] += 1
        return {"ticker": ticker, "trailing_pe": 35.1}

    for ticker in ["MSFT", "AAPL", "MSFT", "GOOGL", "MSFT"]:
        cache.fetch("get_key_ratios", ticker, vendor)
    print(f"    5 fetches -> {vendor_calls['n']} vendor calls "
          f"({cache.hits} cache hits, {cache.misses} misses)\n")

    print("3) cascade over a 500-task morning triage (80 hard tasks escalate)")
    tasks = [{"input_tokens": 1500, "output_tokens": 150,
              "slm_confidence": 0.6 if i < 80 else 0.95} for i in range(500)]
    report = run_cascade(tasks)
    print(f"    {report['tasks']} tasks, {report['escalated']} escalated to gpt-4o")
    print(f"    blended: ${report['blended_cost_usd']}   "
          f"all-gpt-4o: ${report['gpt4o_only_cost_usd']}   "
          f"saving: {report['saving_pct']}%\n")

    print("4) fan-out sized to a rate limit (40 names, 0.2s each, simulated)")

    async def analyse_one(ticker: str) -> str:
        await asyncio.sleep(0.2)                       # stands in for one ~10s agent run
        return "FAIRLY_VALUED"

    names = [f"NAME{i:02d}" for i in range(40)]
    for width in (1, 8, 40):
        secs = asyncio.run(fan_out(names, analyse_one, max_concurrent=width))
        print(f"    concurrency {width:>2}: {secs:5.2f}s wall-clock")
    print("    -> serial is 40x the per-name time; width 8 is 5x faster; width 40 is")
    print("       fastest here but in production would exceed the rate limit and throttle")


if __name__ == "__main__":
    main()
