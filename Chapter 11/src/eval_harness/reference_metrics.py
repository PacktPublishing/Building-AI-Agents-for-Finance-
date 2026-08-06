"""Reference-based metrics: structured outputs compared against gold labels.

Two metrics live here:
- risk_tier_accuracy — exact match on the four-class risk classification
- sanctions_recall — recall on the binary "is this a confirmed match?" label

Both report a Wilson 95% confidence interval alongside the point estimate
because regulators care about uncertainty around the metric, not just the
point value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models import EvaluationResult, KYCAgentOutput, KYCCase


def risk_tier_accuracy(results: Iterable[EvaluationResult]) -> tuple[float, tuple[float, float], int]:
    items = list(results)
    correct = sum(1 for r in items if r.output.risk_tier == r.expected.risk_tier)
    n = len(items)
    if n == 0:
        return 0.0, (0.0, 0.0), 0
    p = correct / n
    return p, _wilson_ci(correct, n), n


def sanctions_recall(results: Iterable[EvaluationResult]) -> tuple[float, tuple[float, float], int]:
    """Recall on confirmed sanctions matches.

    Note: this is RECALL, not accuracy. Failing to match a true sanctions
    case is a catastrophic failure; matching a non-sanctions case is also
    bad but tracked separately as false-positive rate.
    """
    confirmed = [r for r in results if r.expected.sanctions_match]
    if not confirmed:
        return 1.0, (1.0, 1.0), 0
    detected = sum(1 for r in confirmed if r.output.sanctions_match)
    return detected / len(confirmed), _wilson_ci(detected, len(confirmed)), len(confirmed)


def score_case(case: KYCCase, output: KYCAgentOutput) -> dict[str, float]:
    """Per-case reference-based scores. Aggregated upstream."""
    return {
        "risk_tier_correct": float(output.risk_tier == case.expected.risk_tier),
        "sanctions_match_correct": float(output.sanctions_match == case.expected.sanctions_match),
    }


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion.

    Preferred over the normal-approximation interval at small n or extreme p.
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
