"""Trajectory metrics: did the agent take the right path?

Required-tool coverage is a catastrophic-severity check. A KYC trace that
omits sanctions screening is a regulatory event regardless of what the
final risk tier looks like.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models import KYCAgentOutput, KYCCase


def required_tools_invoked(case: KYCCase, output: KYCAgentOutput) -> bool:
    """All required tools must appear in the trace, in the expected order."""
    actual = [tc.tool_name for tc in output.tool_calls]
    expected = case.expected.required_tools
    if not _is_subsequence(expected, actual):
        return False
    return True


def loop_count(output: KYCAgentOutput, threshold: int = 3) -> int:
    """Number of tools invoked more than `threshold` times with same arguments.

    Pathological loops are a cost driver and a quality signal. Threshold
    is configurable because legitimate retries can occur.
    """
    counts: dict[tuple[str, str], int] = {}
    for tc in output.tool_calls:
        key = (tc.tool_name, str(sorted(tc.arguments.items())))
        counts[key] = counts.get(key, 0) + 1
    return sum(1 for v in counts.values() if v > threshold)


def coverage_rate(cases_and_outputs: Iterable[tuple[KYCCase, KYCAgentOutput]]) -> float:
    items = list(cases_and_outputs)
    if not items:
        return 0.0
    passed = sum(1 for c, o in items if required_tools_invoked(c, o))
    return passed / len(items)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Check that every element of `needle` appears in `haystack` in order.

    Allows extra tool calls in between — we require ordering, not strict
    adjacency.
    """
    it = iter(haystack)
    return all(any(item == h for h in it) for item in needle)
