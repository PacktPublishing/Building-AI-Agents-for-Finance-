"""Production trace sampling from Langfuse.

Pulls a configurable percentage of recent production traces, runs the
LLM-judge on each, and emits judge scores for drift comparison against
the offline baseline.

Stub: real implementation depends on the langfuse client and a stable
production deployment. The lab provides this as a structural placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TraceSample:
    trace_id: str
    captured_at: datetime
    reasoning_chain: str
    tool_results: list[dict[str, object]]
    risk_tier: str
    judge_score: float | None = None


def sample_recent_traces(
    hours: int = 24, sample_rate: float = 0.05
) -> list[TraceSample]:
    """Sample recent production traces. Real impl uses langfuse.fetch_traces."""
    raise NotImplementedError(
        "Production trace sampling requires a configured Langfuse instance "
        "and a deployed agent. See the lab README for setup."
    )
