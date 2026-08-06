"""Efficiency metrics: cost, latency, tokens.

These are the metrics that determine whether the harness can be afforded
in production. Latency is reported at p50/p95/p99 because averages hide
SLA-breaking long tails.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..models import KYCAgentOutput


def latency_percentiles(outputs: Iterable[KYCAgentOutput]) -> dict[str, float]:
    """Return p50, p95, p99 latency in milliseconds."""
    items = list(outputs)
    if not items:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array([o.latency_ms for o in items], dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def cost_per_case(outputs: Iterable[KYCAgentOutput]) -> float:
    items = list(outputs)
    if not items:
        return 0.0
    return sum(o.cost_usd for o in items) / len(items)


def avg_tool_calls(outputs: Iterable[KYCAgentOutput]) -> float:
    items = list(outputs)
    if not items:
        return 0.0
    return sum(len(o.tool_calls) for o in items) / len(items)


def token_usage(outputs: Iterable[KYCAgentOutput]) -> dict[str, float]:
    items = list(outputs)
    if not items:
        return {"avg_input": 0.0, "avg_output": 0.0}
    return {
        "avg_input": sum(o.tokens_input for o in items) / len(items),
        "avg_output": sum(o.tokens_output for o in items) / len(items),
    }
