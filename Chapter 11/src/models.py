"""Pydantic schemas for the KYC evaluation harness.

Every artifact that flows through the harness is typed here:
- KYCCase / KYCExpectedOutcome — input + gold answer
- KYCAgentOutput / KYCToolCall — what the agent produces
- EvaluationResult / Scorecard — what the harness produces

Keeping these in one file makes the contract between agent, metrics, and
scorecard inspectable in a single place.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskTier = Literal["LOW", "MEDIUM", "HIGH", "DECLINE"]
Severity = Literal["catastrophic", "severe", "significant"]
Segment = Literal["core", "sanctions", "adversarial", "disparate_impact", "regression"]


class CustomerProfile(BaseModel):
    full_name: str
    date_of_birth: str  # ISO 8601
    jurisdiction: str  # ISO 3166-1 alpha-2
    occupation: str | None = None
    free_text_statement: str | None = None  # adversarial inputs land here
    demographic_metadata: dict[str, str] = Field(default_factory=dict)


class DocumentReference(BaseModel):
    doc_type: Literal["passport", "drivers_license", "national_id", "utility_bill"]
    issuing_jurisdiction: str
    document_id: str
    image_path: str | None = None


class KYCExpectedOutcome(BaseModel):
    risk_tier: RiskTier
    sanctions_match: bool
    required_tools: list[str] = Field(
        default_factory=lambda: [
            "intake.parse",
            "documents.verify",
            "sanctions.screen",
            "media.search",
            "risk.score",
            "decision.adjudicate",
            "compliance.validate",
        ]
    )
    rationale: str
    severity_if_wrong: Severity = "severe"


class CaseMetadata(BaseModel):
    source: Literal["historical", "synthetic", "adversarial", "production_sample"]
    created_at: datetime
    label_version: str = "v1"
    labelers: list[str] = Field(default_factory=list)
    label_kappa: float | None = None  # inter-rater reliability, if multi-labeled


class KYCCase(BaseModel):
    case_id: str
    segment: Segment
    customer: CustomerProfile
    documents: list[DocumentReference] = Field(default_factory=list)
    expected: KYCExpectedOutcome
    metadata: CaseMetadata


class KYCToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object]
    latency_ms: int
    started_at: datetime


class KYCAgentOutput(BaseModel):
    case_id: str
    risk_tier: RiskTier
    sanctions_match: bool
    reasoning_chain: str  # free-form text justification
    tool_calls: list[KYCToolCall]
    audit_log_lines: list[str] = Field(default_factory=list)
    latency_ms: int
    cost_usd: float
    tokens_input: int = 0
    tokens_output: int = 0


class MetricResult(BaseModel):
    metric_name: str
    dimension: str
    methodology: str
    value: float  # 0.0 to 1.0 (or 100 for percentages — normalized in scorecard)
    threshold: float
    passed: bool
    inverted: bool = False  # raw-valued metric where lower is better
    severity: Severity
    weight: float
    notes: str = ""
    detail: dict[str, object] = Field(default_factory=dict)


class Scorecard(BaseModel):
    agent_version: str
    dataset_version: str
    methodology_version: str
    executed_at: datetime
    executor: str
    segment_filter: str = "all"
    metrics: list[MetricResult]
    weighted_score: float
    catastrophic_passes: bool
    ship_decision: Literal["SHIP", "BLOCK"]

    def block_reasons(self) -> list[str]:
        # A metric that never ran fails closed; say so instead of
        # reporting a misleading literal 0.000.
        return [
            (f"{m.metric_name} ({m.severity}): NOT MEASURED in this run "
             f"— fails closed"
             if m.notes and m.notes.startswith("NOT MEASURED")
             else f"{m.metric_name} ({m.severity}): {m.value:.3f} "
                  f"vs threshold {m.threshold:.3f}")
            for m in self.metrics
            if not m.passed and m.severity == "catastrophic"
        ]


class EvaluationResult(BaseModel):
    """Per-case result, separate from per-metric aggregate."""

    case_id: str
    output: KYCAgentOutput
    expected: KYCExpectedOutcome
    metric_values: dict[str, float]  # metric_name -> value
    metric_details: dict[str, dict[str, object]] = Field(default_factory=dict)
