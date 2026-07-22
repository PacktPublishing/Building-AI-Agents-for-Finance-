"""lab1_observability.py : Observing the fundamental-analysis workflow in production.

The workflow runs on the OpenAI Agents SDK, which already emits a trace of spans. In this
lab we do NOT build a tracer; we attach a custom TracingProcessor that captures the SDK's
spans, write them as JSONL, and turn them into the two things operations needs :  1- a trace
tree and 2- per-run metrics; then detect a drift/incident signal when we inject a failure.

Run the whole lab:  python lab1_observability.py
Run one step:       python -c "from lab1_observability import step_4_render_trace; step_4_render_trace()"

The JSONL span format is deliberately backend-neutral; graduating to a real backend
is adding a processor, not a rewrite.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from collections import Counter

from agents.tracing import TracingProcessor, set_trace_processors

import fa_workflow as fa
from tracing import (cost_usd, duration_ms, render_sdk_trace,
                     tool_call_spans, model_call_spans)

if getattr(sys.stdout, "encoding", "").lower() != "utf-8":  # idempotent: wrap once
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(ROOT, "results", "traces")
RUNS_DIR = os.path.join(ROOT, "results", "lab_runs")

WATCHLIST = ["MSFT", "AAPL", "GOOGL", "JPM", "KO"]

# -- Saving all results in a local folder --------------------------------------------------
def _report(name: str, body: str) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{name}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"  -> wrote {os.path.relpath(path, ROOT)}")


# -- The trace processor ------------------------------------------------------------------
class JsonlTraceProcessor(TracingProcessor):
    """Capture the SDK's spans for one run, write them to results/traces/<trace_id>.jsonl,
    and keep them so the later steps can render a tree and roll up metrics.

    The SDK calls on_span_end for every span (agent, turn, model response, tool call) and
    on_trace_end when the whole run finishes. We collect on end and flush on trace end."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.spans: list = []          # spans of the most recent trace
        self._buffer: list = []
        self._trace_id: str | None = None

    def on_trace_start(self, trace):
        self._buffer = []
        self._trace_id = trace.trace_id

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        self._buffer.append(span)

    def on_trace_end(self, trace):
        self.spans = list(self._buffer)
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{trace.trace_id}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for s in self.spans:
                fh.write(json.dumps({
                    "span_id": s.span_id, "parent_id": s.parent_id,
                    "type": s.span_data.type,
                    "name": getattr(s.span_data, "name", None),
                    "started_at": s.started_at, "ended_at": s.ended_at,
                    "duration_ms": duration_ms(s.started_at, s.ended_at),
                    "error": s.error,
                }) + "\n")

    def shutdown(self):
        pass

    def force_flush(self):
        pass


# One processor instance, registered as the ONLY trace processor. This also replaces the
# SDK's default processor, so spans go to us and are not uploaded to OpenAI's trace backend.
_PROCESSOR = JsonlTraceProcessor(TRACES_DIR)
set_trace_processors([_PROCESSOR])

def _run_one(ticker: str, tool_patch=None) -> dict:
    """Run the workflow on one ticker under the trace processor and roll the trace up into
    per-run metrics. tool_patch lets Step 5 inject a failing tool."""
    saved = None
    if tool_patch is not None:
        saved, fa._fetch_ratios = fa._fetch_ratios, tool_patch
    try:
        result = asyncio.run(fa.run_analysis(ticker))
    finally:
        if saved is not None:
            fa._fetch_ratios = saved

    spans = _PROCESSOR.spans
    usage = result.context_wrapper.usage
    tools = tool_call_spans(spans)
    root = next((s for s in spans if s.parent_id is None), None)
    verdict = fa.parse_trailer(result.final_output, "VERDICT")
    return {
        "ticker": ticker,
        "verdict": verdict,
        "model_calls": len(model_call_spans(spans)),
        "tool_calls": len(tools),
        "tool_errors": sum(1 for s in tools if s.error is not None),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd(fa.SLM_MODEL, usage.input_tokens, usage.output_tokens),
        "latency_ms": duration_ms(root.started_at, root.ended_at) if root else 0.0,
        "_spans": spans,
    }


# -- Step 1: the uninstrumented baseline --------------------------------------------------
def step_1_baseline(ticker: str = "MSFT") -> None:
    """Run the workflow and print only its verdict. That single line is all the prototype
    gives us: no cost, no latency, no record of which agents or tools ran. We move beyond it."""
    result = asyncio.run(fa.run_analysis(ticker))
    line = f"[{ticker}] verdict={fa.parse_trailer(result.final_output, 'VERDICT')}"
    print(line)
    _report("oai_step_1_obs_baseline", f"# Step 1 -- uninstrumented baseline\n\n```\n{line}\n```\n") # oai: OpenAI


# -- Steps 2-3: instrumented run + per-run metrics ----------------------------------------
def step_2_3_metrics(ticker: str = "MSFT") -> dict:
    """With the trace processor attached, one run now yields per-run metrics the
    bare verdict never could: model-call count, tool-call count, tokens, cost, latency."""
    m = _run_one(ticker)
    m.pop("_spans")
    lines = [f"{k}: {v}" for k, v in m.items()]
    print("\n".join(lines))
    _report("oai_step_2_3_obs_metrics",
            "# Steps 2-3 -- trace processor + per-run metrics\n\n```\n" + "\n".join(lines) + "\n```\n")
    return m


# -- Step 4: render the SDK's trace as a tree ---------------------------------------------
def step_4_render_trace(ticker: str = "MSFT") -> str:
    """Render the captured SDK spans as a tree : orchestrator -> sub-agent as a tool -> tool call
    -> data tool. The first view an on-call engineer opens."""
    _run_one(ticker)
    tree = render_sdk_trace(_PROCESSOR.spans)
    print(tree)
    _report("oai_step_4_obs_render_trace", f"# Step 4 -- trace tree for {ticker}\n\n```\n{tree}\n```\n")
    return tree


# -- Step 5: metrics rollup over a watchlist ----------------------------------------------
def step_5_metrics_rollup(watchlist: list[str] | None = None) -> list[dict]:
    """Run the watchlist and aggregate per-run metrics into the daily operational readout."""
    watchlist = watchlist or WATCHLIST
    rows = []
    for t in watchlist:
        m = _run_one(t)
        m.pop("_spans")
        rows.append(m)

    header = f"{'ticker':<7}{'verdict':<22}{'llm':<5}{'tools':<6}{'errs':<5}{'cost$':<10}{'ms':<7}"
    body = [header, "-" * len(header)]
    for r in rows:
        body.append(f"{r['ticker']:<7}{r['verdict']:<22}{r['model_calls']:<5}"
                    f"{r['tool_calls']:<6}{r['tool_errors']:<5}{r['cost_usd']:<10.5f}"
                    f"{r['latency_ms']:<7.0f}")
    total = sum(r["cost_usd"] for r in rows)
    dist = Counter(r["verdict"] for r in rows)
    body += ["", f"total cost: ${total:.5f}   verdict distribution: {dict(dist)}"]
    table = "\n".join(body)
    print(table)
    _report("oai_step_5_obs_metrics_rollup", f"# Step 5 -- metrics rollup\n\n```\n{table}\n```\n")
    return rows


# -- Step 6: drift / incident signals -----------------------------------------------------
def _flaky_ratios(ticker: str):
    """Stand-in for a data-vendor outage: the ratios fetch raises. The SDK records the error
    on the tool's span, so the incident detector has a real signal to catch."""
    raise ConnectionError("ratios vendor timed out")


def step_6_drift_signals(watchlist: list[str] | None = None) -> dict:
    """Compare a healthy baseline window against a current window in which the ratios tool
    fails. Alert on any signal crossing its threshold: tool-error rate or a jump in the
    INSUFFICIENT_EVIDENCE share. This is telemetry (did something break?), not evaluation
    (was a verdict right?). Scoring verdict content belongs to the evaluation chapter."""
    watchlist = watchlist or WATCHLIST
    baseline = [_run_one(t) for t in watchlist]
    current = [_run_one(t, tool_patch=_flaky_ratios) for t in watchlist]
    for r in baseline + current:
        r.pop("_spans")

    def err_rate(rows):
        calls = sum(r["tool_calls"] for r in rows)
        errs = sum(r["tool_errors"] for r in rows)
        return round(errs / calls, 3) if calls else 0.0

    def insufficient_share(rows):
        n = sum(1 for r in rows if r["verdict"] == "INSUFFICIENT_EVIDENCE")
        return round(n / len(rows), 3)

    signals = {
        "tool_error_rate": (err_rate(baseline), err_rate(current), 0.10),
        "insufficient_share": (insufficient_share(baseline), insufficient_share(current), 0.25),
        "avg_cost_usd": (round(sum(r["cost_usd"] for r in baseline) / len(baseline), 6),
                         round(sum(r["cost_usd"] for r in current) / len(current), 6), None),
    }
    alerts = [f"ALERT {name}: {base} -> {cur} (threshold {thr})"
              for name, (base, cur, thr) in signals.items()
              if thr is not None and cur >= thr and cur > base]

    lines = [f"{name:<20} baseline={b}  current={c}  threshold={t}"
             for name, (b, c, t) in signals.items()] + [""] + (alerts or ["no alerts"])
    text = "\n".join(lines)
    print(text)
    _report("oai_step_6_obs_drift_signals", f"# Step 6 -- drift / incident signals\n\n```\n{text}\n```\n")
    return {"signals": signals, "alerts": alerts}


def main() -> None:
    print("== Step 1: uninstrumented baseline ==");      step_1_baseline()
    print("\n== Steps 2-3: processor + metrics ==");     step_2_3_metrics()
    print("\n== Step 4: trace tree ==");                 step_4_render_trace()
    print("\n== Step 5: metrics rollup ==");             step_5_metrics_rollup()
    print("\n== Step 6: drift / incident signals ==");   step_6_drift_signals()


if __name__ == "__main__":
    main()
