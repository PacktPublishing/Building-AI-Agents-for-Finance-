"""Disparate-impact analysis across protected demographic groups.

The metric is statistical, not per-case. We partition outcomes by group,
compute decline rates, and apply two checks:

1. Four-fifths rule: any group's selection rate must be >= 80% of the
   highest group's rate. (EEOC's Uniform Guidelines on Employee Selection
   Procedures, applied here to KYC approval rates.)
2. Chi-square test for independence between group and decision.

Failing either is a catastrophic-severity event.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

from ..models import EvaluationResult


def decline_rates_by_group(results: Iterable[EvaluationResult]) -> dict[str, float]:
    """Compute decline rate per demographic group.

    Group is read from case.customer.demographic_metadata['group']. Cases
    without a group label are excluded from this analysis.
    """
    items = list(results)
    by_group: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in items:
        # Recovering the case from the result requires the harness to pass
        # cases through; for now we look it up in metric_details.
        group = r.metric_details.get("disparate_impact", {}).get("group")
        if isinstance(group, str):
            by_group[group].append(r)
    return {
        g: sum(1 for r in rs if r.output.risk_tier == "DECLINE") / len(rs)
        for g, rs in by_group.items()
        if rs
    }


def four_fifths_rule(rates: dict[str, float]) -> tuple[bool, str | None]:
    """Returns (passes, failing_group_or_None).

    Decline rates are inverse-coded selection rates; we apply the four-fifths
    rule to (1 - decline_rate) which is the approval/selection rate.
    """
    if len(rates) < 2:
        return True, None
    selection = {g: 1.0 - dr for g, dr in rates.items()}
    max_rate = max(selection.values())
    if max_rate == 0:
        return True, None
    for g, sel in selection.items():
        if sel / max_rate < 0.8:
            return False, g
    return True, None


def chi_square_test(results: Iterable[EvaluationResult]) -> tuple[float, float]:
    """Returns (chi2_statistic, p_value) for group × decision independence.

    Guards: with a single group, a single observed decision, or an empty
    table there is no variation to test, so we return (0.0, 1.0). With
    small samples (any expected cell below 5) the asymptotic chi-square
    approximation is unreliable; for 2×2 tables we switch to Fisher's
    exact test, and larger sparse tables keep chi-square but the caller
    should treat the p-value as a screen, not proof.
    """
    by_group: dict[str, list[str]] = defaultdict(list)
    for r in results:
        group = r.metric_details.get("disparate_impact", {}).get("group")
        if isinstance(group, str):
            by_group[group].append(r.output.risk_tier)
    if len(by_group) < 2:
        return 0.0, 1.0
    groups = sorted(by_group)
    tiers = sorted({t for ts in by_group.values() for t in ts})
    if len(tiers) < 2:
        return 0.0, 1.0
    contingency = np.array(
        [[by_group[g].count(t) for t in tiers] for g in groups], dtype=float
    )
    if contingency.sum() == 0:
        return 0.0, 1.0
    expected = (contingency.sum(axis=1, keepdims=True)
                @ contingency.sum(axis=0, keepdims=True)) / contingency.sum()
    if contingency.shape == (2, 2) and expected.min() < 5:
        _, p = fisher_exact(contingency)
        return 0.0, float(p)
    chi2, p, _, _ = chi2_contingency(contingency)
    return float(chi2), float(p)


def evaluate(results: Iterable[EvaluationResult]) -> dict[str, object]:
    """Combined disparate-impact evaluation. Returns scorecard-ready dict."""
    items = list(results)
    rates = decline_rates_by_group(items)
    passes_4_5, failing = four_fifths_rule(rates)
    chi2, p = chi_square_test(items)
    return {
        "rates_by_group": rates,
        "four_fifths_passes": passes_4_5,
        "failing_group": failing,
        "chi2": chi2,
        "p_value": p,
        "stat_significant_disparity": p < 0.05,
        "passes": passes_4_5 and p >= 0.05,
    }
