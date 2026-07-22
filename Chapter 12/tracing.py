"""tracing.py : Shared observability helpers for the labs.

The OpenAI Agents SDK already produces a trace: a tree of spans (a task span at the root,
agent spans, turn spans, model-response spans, and function/tool spans). We do not
re-implement tracing; instead we (a) capture those spans through a custom trace processor
(see lab1_observability.py) and (b) turn them into the two things operations actually
needs, 1- a readable trace tree and 2- a per-run cost,  with the small helpers here.

Keeping this layer thin is the point: production teams ship the same spans to a backend
such as MLflow, Langfuse, or an OpenTelemetry collector.
Everything here works on the SDK's own Span objects, so nothing is bespoke.
"""

from __future__ import annotations

from datetime import datetime

# Published OpenAI per-1M-token prices for the tiers this workflow uses.
PRICE_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),     # the SLM tier
    "gpt-4o": (2.50, 10.00),         # the escalation tier
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of a run's model calls -- the line item of the agent's P&L."""
    in_price, out_price = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return round((input_tokens * in_price + output_tokens * out_price) / 1_000_000, 6)


def duration_ms(started_at, ended_at) -> float:
    """Milliseconds between two ISO-8601 timestamps (the SDK's span start/end format)."""
    if not started_at or not ended_at:
        return 0.0
    a = datetime.fromisoformat(started_at)
    b = datetime.fromisoformat(ended_at)
    return round((b - a).total_seconds() * 1000, 1)


# Span types the SDK emits; we keep the ones that matter in a trace tree and drop the noise.
_INTERESTING = {"agent", "function", "response"}
# Leaf data tools (as opposed to sub-agents-exposed-as-tools, whose names differ).
_DATA_TOOLS = {"get_key_ratios", "get_recent_news"}


def _label(span) -> str:
    """A short label for a span: its type plus a name where the SDK provides one."""
    sd = span.span_data
    stype = sd.type
    name = getattr(sd, "name", None)
    if stype == "agent":
        return f"agent:{name}"
    if stype == "function":
        return f"tool:{name}"
    if stype == "response":
        return "llm.call"
    return stype


def render_sdk_trace(spans: list) -> str:
    """Render the SDK's spans as an indented tree -- the view an on-call engineer opens first.

    We show agent, tool, and model-call spans (dropping the SDK's internal task/turn wrapper
    spans) so the tree reads as: orchestrator -> sub-agent tool -> sub-agent -> data tool."""
    by_parent: dict = {}
    for s in spans:
        by_parent.setdefault(s.parent_id, []).append(s)
    for children in by_parent.values():
        children.sort(key=lambda s: s.started_at or "")

    lines: list[str] = []

    def walk(parent_id, depth: int) -> None:
        for s in by_parent.get(parent_id, []):
            if s.span_data.type in _INTERESTING:
                mark = "x" if s.error else " "
                dur = duration_ms(s.started_at, s.ended_at)
                lines.append(f"{'  ' * depth}[{mark}] {_label(s)} ({dur:.0f} ms)")
                walk(s.span_id, depth + 1)
            else:
                # Skip the wrapper span but keep walking so its children attach one level up.
                walk(s.span_id, depth)

    walk(None, 0)
    return "\n".join(lines)


def tool_call_spans(spans: list) -> list:
    """The leaf data-tool calls (get_key_ratios / get_recent_news), for counting and errors."""
    return [s for s in spans
            if s.span_data.type == "function"
            and getattr(s.span_data, "name", None) in _DATA_TOOLS]


def model_call_spans(spans: list) -> list:
    """The model-response spans -- how many model turns the run took."""
    return [s for s in spans if s.span_data.type == "response"]
