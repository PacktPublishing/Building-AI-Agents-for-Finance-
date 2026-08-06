"""LLM-judge calibration against human-labeled samples.

Workflow:
    1. Load human labels from data/kyc_eval/calibration_set.jsonl
    2. Compute Cohen's kappa between primary labelers (must be >= 0.7
       before any judge work — if humans disagree, the rubric is
       ambiguous)
    3. Run the LLM-judge on the same cases
    4. Compute kappa between judge and human consensus
    5. Iterate the rubric until judge-vs-human kappa >= 0.7
    6. Freeze the rubric, judge model, calibration set, and kappa as
       a versioned regulatory artifact
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ..common import DATA_DIR
from ..eval_harness.llm_judge_metrics import LABEL_TO_SCORE, GroundingLabel

CALIBRATION_PATH = DATA_DIR / "calibration_set.jsonl"


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters on the same set of items.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative
    for systematic disagreement.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater sequences must have equal length")
    n = len(rater_a)
    if n == 0:
        return 0.0
    categories = sorted(set(rater_a) | set(rater_b))
    observed = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b) / n
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def load_calibration_set(path: Path = CALIBRATION_PATH) -> list[dict[str, object]]:
    """Each line: {case_id, reasoning, tool_results, label_a, label_b, label_consensus}."""
    if not path.exists():
        return []
    items: list[dict[str, object]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_calibration(
    judge_predictions: Sequence[GroundingLabel],
    human_consensus: Sequence[GroundingLabel],
) -> dict[str, float]:
    """Score a candidate judge against the human consensus labels."""
    kappa = cohens_kappa(list(judge_predictions), list(human_consensus))
    score_corr = _score_correlation(judge_predictions, human_consensus)
    return {
        "cohens_kappa": kappa,
        "score_correlation": score_corr,
        "calibrated": kappa >= 0.7,
    }


def inter_rater_reliability(items: list[dict[str, object]]) -> float:
    """Kappa between primary human labelers. Run this BEFORE judge work."""
    if not items or "label_a" not in items[0] or "label_b" not in items[0]:
        return 0.0
    a = [str(it["label_a"]) for it in items]
    b = [str(it["label_b"]) for it in items]
    return cohens_kappa(a, b)


def _score_correlation(
    predicted: Sequence[GroundingLabel], target: Sequence[GroundingLabel]
) -> float:
    """Pearson correlation on the numeric LABEL_TO_SCORE projection."""
    if len(predicted) != len(target) or not predicted:
        return 0.0
    p = [LABEL_TO_SCORE[label] for label in predicted]
    t = [LABEL_TO_SCORE[label] for label in target]
    n = len(p)
    mean_p = sum(p) / n
    mean_t = sum(t) / n
    num = sum((pi - mean_p) * (ti - mean_t) for pi, ti in zip(p, t, strict=True))
    denom_p = (sum((pi - mean_p) ** 2 for pi in p)) ** 0.5
    denom_t = (sum((ti - mean_t) ** 2 for ti in t)) ** 0.5
    if denom_p == 0 or denom_t == 0:
        return 0.0
    return num / (denom_p * denom_t)
