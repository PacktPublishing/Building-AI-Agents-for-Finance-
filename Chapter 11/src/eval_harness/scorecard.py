"""Scorecard composer.

Reads scorecard.yaml, runs each configured metric against the appropriate
dataset segment, aggregates per-metric results into a Scorecard artifact,
and produces both a JSON file (for tooling) and a Markdown summary (for
humans). Catastrophic-severity threshold failures override the weighted
score and force a BLOCK ship decision.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..common import AGENT_VERSION, DATASET_VERSION, METHODOLOGY_VERSION
from ..models import EvaluationResult, MetricResult, Scorecard, Severity

CONFIG_PATH = Path(__file__).parent / "scorecard.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def compose(
    metric_values: dict[str, float],
    metric_details: dict[str, dict[str, Any]] | None = None,
    config_path: Path = CONFIG_PATH,
    executor: str = "ci",
) -> Scorecard:
    """Build a Scorecard from raw metric values.

    `metric_values` maps metric name (matching scorecard.yaml `name`) to
    a 0..1 score (or a raw value for inverted metrics like latency_ms).
    """
    config = load_config(config_path)
    details = metric_details or {}

    metrics: list[MetricResult] = []
    for spec in config["metrics"]:
        name: str = spec["name"]
        if name not in metric_values:
            # Missing metric is itself a failure — we cannot ship without
            # measuring something we promised the committee we would.
            metrics.append(_unmeasured(spec))
            continue
        metrics.append(_evaluate_metric(spec, metric_values[name], details.get(name, {})))

    weighted = _weighted_score(metrics)
    catastrophic_passes = all(
        m.passed for m in metrics if m.severity == "catastrophic"
    )
    ship = "SHIP" if catastrophic_passes and weighted >= 0.85 else "BLOCK"

    return Scorecard(
        agent_version=AGENT_VERSION,
        dataset_version=DATASET_VERSION,
        methodology_version=METHODOLOGY_VERSION,
        executed_at=datetime.now(timezone.utc),
        executor=executor,
        metrics=metrics,
        weighted_score=weighted,
        catastrophic_passes=catastrophic_passes,
        ship_decision=ship,  # type: ignore[arg-type]
    )


def _evaluate_metric(spec: dict[str, Any], value: float, detail: dict[str, Any]) -> MetricResult:
    threshold = float(spec["threshold"])
    inverted = spec.get("inverted", False)
    passed = value <= threshold if inverted else value >= threshold

    severity: Severity = spec["severity"]
    return MetricResult(
        metric_name=spec["name"],
        dimension=spec["dimension"],
        methodology=spec["methodology"],
        value=float(value),
        threshold=threshold,
        passed=passed,
        inverted=inverted,
        severity=severity,
        weight=float(spec["weight"]),
        notes=spec.get("description", ""),
        detail=detail,
    )


def _unmeasured(spec: dict[str, Any]) -> MetricResult:
    return MetricResult(
        metric_name=spec["name"],
        dimension=spec["dimension"],
        methodology=spec["methodology"],
        value=0.0,
        threshold=float(spec["threshold"]),
        passed=False,
        severity=spec["severity"],
        weight=float(spec["weight"]),
        notes=f"NOT MEASURED — {spec.get('description', '')}",
    )


def _weighted_score(metrics: Iterable[MetricResult]) -> float:
    """Weighted average normalized to 0..1.

    Inverted metrics (latency, cost) contribute their pass-state (1.0 if
    pass, 0.0 if fail) since their numeric values are not in 0..1.
    """
    total_weight = 0.0
    weighted = 0.0
    for m in metrics:
        if m.passed:
            contribution = 1.0
        elif m.inverted:
            # Raw-valued inverted metrics (latency-ms, cost-USD): a failed
            # threshold earns no credit — the raw value is not a 0..1 score.
            contribution = 0.0
        elif 0.0 <= m.value <= 1.0:
            contribution = m.value  # partial credit for in-range scores
        else:
            contribution = 0.0
        weighted += m.weight * contribution
        total_weight += m.weight
    if total_weight == 0:
        return 0.0
    return weighted / total_weight


def write_json(scorecard: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scorecard.model_dump_json(indent=2))


def write_markdown(scorecard: Scorecard, path: Path) -> None:
    """Human-readable scorecard report. Committee one-pager goes first."""
    lines: list[str] = []
    lines.append("# KYC Agent Scorecard\n")
    lines.append(f"**Decision:** `{scorecard.ship_decision}`  ")
    lines.append(f"**Weighted score:** {scorecard.weighted_score:.3f}  ")
    lines.append(f"**Catastrophic checks pass:** {scorecard.catastrophic_passes}  ")
    lines.append(f"**Agent version:** `{scorecard.agent_version}`  ")
    lines.append(f"**Dataset version:** `{scorecard.dataset_version}`  ")
    lines.append(f"**Executed at:** {scorecard.executed_at.isoformat()}  \n")

    if not scorecard.catastrophic_passes:
        lines.append("## Catastrophic-severity failures (block reasons)\n")
        for reason in scorecard.block_reasons():
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("## Per-metric results\n")
    lines.append("| Metric | Dimension | Value | Threshold | Severity | Pass |")
    lines.append("|--------|-----------|-------|-----------|----------|------|")
    for m in scorecard.metrics:
        check = "PASS" if m.passed else "FAIL"
        lines.append(
            f"| {m.metric_name} | {m.dimension} | {m.value:.3f} | "
            f"{m.threshold:.3f} | {m.severity} | {check} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def from_results(results: list[EvaluationResult]) -> Scorecard:
    """Convenience: build a scorecard from per-case EvaluationResult list."""
    if not results:
        return compose({}, {})
    aggregated: dict[str, float] = {}
    # The harness run() already populates per-metric values into each result;
    # here we just average them. Real aggregation lives in run.py.
    for name in results[0].metric_values:
        aggregated[name] = sum(r.metric_values.get(name, 0.0) for r in results) / len(results)
    return compose(aggregated, {})
