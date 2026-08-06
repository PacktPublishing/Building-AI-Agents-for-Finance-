"""Model risk report generator.

Renders a model-risk-committee-ready Markdown document from a Scorecard.
The committee reviews the one-page executive summary first; technical
sections follow.

For PDF rendering, pipe the output through pandoc:
    python -m src.reporting.model_risk_report scorecard.json | \
        pandoc -o model_risk_report.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..models import Scorecard


def render(scorecard: Scorecard) -> str:
    blocks: list[str] = []
    blocks.append(_executive_summary(scorecard))
    blocks.append(_scope_and_methodology(scorecard))
    blocks.append(_per_metric_section(scorecard))
    blocks.append(_aggregate_section(scorecard))
    blocks.append(_known_limitations())
    blocks.append(_monitoring_plan())
    blocks.append(_signoff_section())
    return "\n\n".join(blocks)


def _executive_summary(s: Scorecard) -> str:
    decision_marker = "[BLOCK]" if s.ship_decision == "BLOCK" else "[SHIP]"
    lines = [
        "# Model Risk Report — KYC Agent Evaluation",
        "",
        "## Executive Summary",
        "",
        f"**Ship decision:** `{decision_marker} {s.ship_decision}`",
        f"**Weighted score:** {s.weighted_score:.3f}",
        f"**Catastrophic-severity checks:** {'PASS' if s.catastrophic_passes else 'FAIL'}",
        f"**Agent version:** `{s.agent_version}`",
        f"**Dataset version:** `{s.dataset_version}`",
        f"**Executed at:** {s.executed_at.isoformat()}",
    ]
    if not s.catastrophic_passes:
        lines.append("")
        lines.append("**Block reasons (catastrophic-severity threshold failures):**")
        for reason in s.block_reasons():
            lines.append(f"- {reason}")
    return "\n".join(lines)


def _scope_and_methodology(s: Scorecard) -> str:
    return (
        "## 1. Scope and Methodology\n\n"
        f"This report covers agent version `{s.agent_version}` evaluated against "
        f"dataset version `{s.dataset_version}` using methodology version "
        f"`{s.methodology_version}`. All metrics are defined in scorecard.yaml. "
        "Catastrophic-severity metrics are pass/fail regardless of weighted score; "
        "any failure blocks production deployment."
    )


def _per_metric_section(s: Scorecard) -> str:
    lines = ["## 2. Per-Metric Results", "",
             "| Metric | Dimension | Method | Value | Threshold | Severity | Pass |",
             "|--------|-----------|--------|-------|-----------|----------|------|"]
    for m in s.metrics:
        check = "PASS" if m.passed else "FAIL"
        lines.append(
            f"| {m.metric_name} | {m.dimension} | {m.methodology} | "
            f"{m.value:.3f} | {m.threshold:.3f} | {m.severity} | {check} |"
        )
    return "\n".join(lines)


def _aggregate_section(s: Scorecard) -> str:
    return (
        "## 3. Aggregate Scorecard\n\n"
        f"Weighted score: **{s.weighted_score:.3f}**. "
        f"Catastrophic checks: **{'PASS' if s.catastrophic_passes else 'FAIL'}**. "
        f"Final ship decision: **{s.ship_decision}**."
    )


def _known_limitations() -> str:
    return (
        "## 4. Known Limitations\n\n"
        "- The harness treats the agent as a black box; tool-level faults "
        "internal to the pipeline may not surface unless they affect the "
        "trace or final output.\n"
        "- LLM-judge metrics depend on the calibration set; recalibrate "
        "quarterly to detect judge-model drift.\n"
        "- Synthetic adversarial cases may not represent every real-world "
        "attack pattern; supplement with red-team campaigns."
    )


def _monitoring_plan() -> str:
    return (
        "## 5. Production Monitoring Plan\n\n"
        "- Trace capture: 100% via Langfuse OpenTelemetry instrumentation.\n"
        "- Online sampling: stratified 5% of traffic, with rare-event oversampling.\n"
        "- Drift detection: hourly KS + PSI + Mann-Whitney against the "
        "offline reference distribution.\n"
        "- Catastrophic-severity metric thresholds re-validated on every "
        "model or prompt promotion."
    )


def _signoff_section() -> str:
    return (
        "## 6. Sign-off\n\n"
        "| Role | Name | Date | Signature |\n"
        "|------|------|------|-----------|\n"
        "| Model Owner (First Line) | | | |\n"
        "| Model Risk Validation (Second Line) | | | |\n"
        "| Internal Audit (Third Line) | | | |\n"
        "| Model Risk Committee Chair | | | |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-risk-report")
    parser.add_argument("scorecard_json", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output Markdown path. Default: stdout.")
    args = parser.parse_args(argv)

    scorecard = Scorecard.model_validate_json(args.scorecard_json.read_text())
    output = render(scorecard)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    else:
        sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
