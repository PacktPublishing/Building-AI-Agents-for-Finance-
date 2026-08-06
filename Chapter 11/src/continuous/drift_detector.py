"""Production drift detection.

Compares two distributions: LLM-judge scores on the offline reference
dataset (baseline) and a rolling sample of recent production traffic
(current). Three tests:

- Kolmogorov-Smirnov two-sample test (statistical significance)
- Population Stability Index (effect size, used in financial risk
  modeling for decades)
- Mann-Whitney U for ordered scores (rank-based)

Any of the three tripping its threshold raises a drift alert. Multiple
tests reduce false alarms vs any single test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu


@dataclass
class DriftResult:
    ks_statistic: float
    ks_p_value: float
    psi: float
    mannwhitney_p: float
    drift_detected: bool
    severity: str  # 'none' | 'warn' | 'alert'

    def explain(self) -> str:
        return (
            f"KS={self.ks_statistic:.3f} (p={self.ks_p_value:.4f}), "
            f"PSI={self.psi:.3f}, MW p={self.mannwhitney_p:.4f}, "
            f"severity={self.severity}"
        )


def detect_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    p_alert: float = 0.01,
) -> DriftResult:
    """Compare current against baseline with KS, PSI, and Mann-Whitney."""
    if len(baseline_scores) == 0 or len(current_scores) == 0:
        return DriftResult(0.0, 1.0, 0.0, 1.0, False, "none")

    ks_stat, ks_p = ks_2samp(baseline_scores, current_scores)
    psi = compute_psi(baseline_scores, current_scores)
    mw_stat, mw_p = mannwhitneyu(baseline_scores, current_scores, alternative="two-sided")

    severity = "none"
    if psi >= psi_alert or ks_p < p_alert or mw_p < p_alert:
        severity = "alert"
    elif psi >= psi_warn:
        severity = "warn"

    return DriftResult(
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        psi=float(psi),
        mannwhitney_p=float(mw_p),
        drift_detected=severity in ("warn", "alert"),
        severity=severity,
    )


def compute_psi(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Population Stability Index.

    PSI = sum over bins of (current_pct - baseline_pct) * ln(current_pct / baseline_pct)

    Industry conventions:
    - PSI < 0.10: no significant change
    - 0.10 ≤ PSI < 0.25: moderate shift, monitor
    - PSI ≥ 0.25: significant shift, investigate
    """
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    baseline_hist, _ = np.histogram(baseline, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)

    epsilon = 1e-6
    baseline_pct = baseline_hist / max(1, baseline_hist.sum()) + epsilon
    current_pct = current_hist / max(1, current_hist.sum()) + epsilon

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
