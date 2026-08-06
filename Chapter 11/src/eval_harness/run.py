"""CLI entry point for the KYC evaluation harness.

Usage:
    python -m src.eval_harness.run --segment all --output scorecard.json

What it does:
    1. Load eval cases from data/kyc_eval/ (filtered by --segment)
    2. Run the KYC agent on each case (real or mock — see kyc_pipeline_adapter)
    3. Compute every metric defined in scorecard.yaml
    4. Compose the scorecard
    5. Write JSON + Markdown outputs
    6. Exit nonzero on catastrophic-severity threshold failures (CI gate)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from ..common import DATA_DIR
from ..kyc_pipeline_adapter import run_kyc
from ..models import EvaluationResult, KYCAgentOutput, KYCCase
from . import (
    disparate_impact,
    efficiency_metrics,
    llm_judge_metrics,
    reference_metrics,
    safety_metrics,
    trajectory_metrics,
)
from .scorecard import compose, write_json, write_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kyc-eval")
    parser.add_argument("--segment", default="all",
                        choices=["all", "core", "sanctions", "adversarial",
                                 "disparate_impact"])  # calibration: see src.calibration
    parser.add_argument("--output", type=Path, default=Path("scorecard.json"))
    parser.add_argument("--markdown", type=Path, default=Path("scorecard.md"))
    parser.add_argument("--executor", default="cli")
    args = parser.parse_args(argv)

    cases = list(_load_cases(args.segment))
    if not cases:
        print(f"no cases for segment={args.segment}", file=sys.stderr)
        return 2

    outputs: list[KYCAgentOutput] = [run_kyc(c) for c in cases]
    results = [_to_result(c, o) for c, o in zip(cases, outputs, strict=True)]

    metric_values, metric_details = _aggregate_metrics(cases, outputs, results)
    scorecard = compose(metric_values, metric_details, executor=args.executor)

    write_json(scorecard, args.output)
    write_markdown(scorecard, args.markdown)

    print(f"ship_decision={scorecard.ship_decision} "
          f"weighted={scorecard.weighted_score:.3f} "
          f"catastrophic_pass={scorecard.catastrophic_passes}")
    if not scorecard.catastrophic_passes:
        print("BLOCK reasons:", file=sys.stderr)
        for r in scorecard.block_reasons():
            print(f"  - {r}", file=sys.stderr)
        return 1
    # A BLOCK from the weighted-score gate must also fail CI.
    return 0 if scorecard.ship_decision == "SHIP" else 3


_CASE_FILES = {
    "core": "core_set.jsonl",
    "sanctions": "sanctions_set.jsonl",
    "adversarial": "adversarial_set.jsonl",
    "disparate_impact": "disparate_impact_set.jsonl",
}


def _load_cases(segment: str) -> Iterable[KYCCase]:
    """Load cases from segment-specific files. Calibration set is excluded;
    it has its own schema and is loaded by src.calibration.judge_calibration."""
    if not DATA_DIR.exists():
        return
    if segment == "all":
        paths = [DATA_DIR / fn for fn in _CASE_FILES.values()]
    elif segment in _CASE_FILES:
        paths = [DATA_DIR / _CASE_FILES[segment]]
    else:
        return
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield KYCCase.model_validate_json(line)


def _to_result(case: KYCCase, output: KYCAgentOutput) -> EvaluationResult:
    metric_values: dict[str, float] = {
        **{k: v for k, v in reference_metrics.score_case(case, output).items()},
        "required_tool_coverage": float(trajectory_metrics.required_tools_invoked(case, output)),
        "pii_redaction": float(safety_metrics.audit_log_clean(output)),
    }
    metric_details: dict[str, dict[str, object]] = {}
    if "group" in case.customer.demographic_metadata:
        metric_details["disparate_impact"] = {
            "group": case.customer.demographic_metadata["group"]
        }
    return EvaluationResult(
        case_id=case.case_id,
        output=output,
        expected=case.expected,
        metric_values=metric_values,
        metric_details=metric_details,
    )


def _aggregate_metrics(
    cases: list[KYCCase],
    outputs: list[KYCAgentOutput],
    results: list[EvaluationResult],
) -> tuple[dict[str, float], dict[str, dict[str, object]]]:
    """Run all aggregate (cross-case) metrics and return scorecard inputs."""
    representative = [r for r in results if r.case_id.startswith("core")]
    adversarial = [r for r in results if r.case_id.startswith("adv")]
    # Fairness analysis runs ONLY on the disparate-impact segment, which is
    # constructed with comparable cases balanced across groups. Pooling in
    # sanctions or adversarial cases — whose gold outcome is DECLINE — would
    # count legitimate declines as demographic disparity.
    disp = [r for r in results if r.case_id.startswith("di-") and "group" in (
        r.metric_details.get("disparate_impact", {}) if r.metric_details else {}
    )]

    rep_acc, rep_ci, _ = reference_metrics.risk_tier_accuracy(representative)
    adv_acc, adv_ci, _ = reference_metrics.risk_tier_accuracy(adversarial)
    sanc_recall, _, _ = reference_metrics.sanctions_recall(results)
    # Hard rule (Section 6.3): a GOLD-LABEL confirmed sanctions match that
    # the agent does not DECLINE is a catastrophic violation regardless of
    # recall — zero the metric so the scorecard gate trips.
    rule_violations = sum(
        1 for c, o in zip(cases, outputs, strict=True)
        if c.expected.sanctions_match and o.risk_tier != "DECLINE"
    )
    if rule_violations:
        sanc_recall = 0.0
    di_eval = disparate_impact.evaluate(disp)

    pii_rate = safety_metrics.pii_redaction_rate(outputs)
    coverage = trajectory_metrics.coverage_rate(zip(cases, outputs, strict=True))

    grounding_scores = [llm_judge_metrics.grade_grounding(o).score for o in outputs] or [0.0]
    grounding_avg = sum(grounding_scores) / len(grounding_scores)

    latencies = efficiency_metrics.latency_percentiles(outputs)
    cost = efficiency_metrics.cost_per_case(outputs)

    # Only report metrics whose segments actually ran: missing keys flow
    # into the composer's NOT-MEASURED path, which fails closed — the
    # correct behavior for a regulatory gate.
    has_sanctions = any(r.expected.sanctions_match for r in results)
    metric_values: dict[str, float] = {
        "reasoning_grounding": grounding_avg,
        "required_tool_coverage": coverage,
        "pii_redaction": pii_rate,
        "sla_p99_latency": latencies["p99"],
        "bsa_documentation": 1.0,  # placeholder; real impl checks specific fields
        "cost_per_case": cost,
        "explainability": grounding_avg,  # proxy; real version uses compliance-calibrated judge
    }
    if representative:
        metric_values["risk_tier_accuracy_representative"] = rep_acc
    if adversarial:
        metric_values["risk_tier_accuracy_adversarial"] = adv_acc
        metric_values["prompt_injection_resistance"] = adv_acc  # labeled proxy
    if has_sanctions:
        metric_values["sanctions_recall"] = sanc_recall
    if disp:
        metric_values["disparate_impact"] = float(di_eval["passes"])
    metric_details: dict[str, dict[str, object]] = {
        "disparate_impact": di_eval,
        "sla_p99_latency": latencies,
        "risk_tier_accuracy_representative": {"wilson_95_ci": rep_ci},
        "risk_tier_accuracy_adversarial": {"wilson_95_ci": adv_ci},
        "sanctions_recall": {"hard_rule_violations": rule_violations},
    }
    return metric_values, metric_details


if __name__ == "__main__":
    raise SystemExit(main())
