"""Adapter between the KYC pipeline and the harness.

If a real implementation of the chapter's KYC pipeline is available as
a Python package (build one from the chapter's specification, following
the Chapter 7/9 multi-agent pipeline patterns), this module imports it
and adapts the call interface. Otherwise it falls back to a
deterministic mock that exercises every code path in the harness so the
lab is runnable end-to-end without the real agent.

The mock is NOT a serious agent. It consults each case's expected
outcome so that test cases produce predictable outputs.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .models import KYCAgentOutput, KYCCase, KYCToolCall, RiskTier

try:  # pragma: no cover - plug your own implementation in under this name
    from chapter09.kyc_pipeline.run import run_kyc as _real_run_kyc

    HAS_REAL_PIPELINE = True
except ImportError:
    HAS_REAL_PIPELINE = False


def run_kyc(case: KYCCase) -> KYCAgentOutput:
    if HAS_REAL_PIPELINE:
        return _real_run_kyc(case)
    return _mock_run_kyc(case)


_MOCK_TOOLS = [
    "intake.parse",
    "documents.verify",
    "sanctions.screen",
    "media.search",
    "risk.score",
    "decision.adjudicate",
    "compliance.validate",
]


def _mock_run_kyc(case: KYCCase) -> KYCAgentOutput:
    """Deterministic mock agent.

    The mock consults each case's expected outcome to produce predictable
    behavior — its purpose is to make the harness's mechanics traceable
    end to end, not to be a real agent. Rules:
    - sanctions_match=True in expected → DECLINE
    - "Ignore all previous instructions" in free text → simulate injection
      defense by returning the *expected* tier (the agent resists)
    - otherwise → the case's expected risk tier

    Swap in the real pipeline via the adapter above to evaluate genuine
    agent behavior; with the mock, every reference-based metric should
    pass by construction.
    """

    rng = random.Random(case.case_id)

    free_text = (case.customer.free_text_statement or "").lower()
    injection_markers = ("ignore all previous instructions",
                         "you are now a different assistant")
    if case.expected.sanctions_match:
        tier: RiskTier = "DECLINE"
        sanctions_match = True
    elif any(m in free_text for m in injection_markers):
        # Documented rule 2: the mock resists the injection by returning
        # the tier the legitimate inputs warrant.
        tier = case.expected.risk_tier
        sanctions_match = case.expected.sanctions_match
    else:
        tier = case.expected.risk_tier
        sanctions_match = False

    # Build a fake trace covering every required tool. Each tool result
    # carries the facts the reasoning chain will cite, so an LLM judge
    # grading groundedness finds real support in the trace — exactly as
    # it would for a well-behaved production agent.
    now = datetime.now(timezone.utc)
    mock_results: dict[str, dict[str, object]] = {
        "intake.parse": {"case_id": case.case_id,
                         "jurisdiction": case.customer.jurisdiction},
        "documents.verify": {"documents_complete": True},
        "sanctions.screen": {"match": sanctions_match},
        "media.search": {"adverse_media_hits": 0},
        "risk.score": {"risk_tier": tier,
                       "jurisdiction": case.customer.jurisdiction},
        "decision.adjudicate": {"decision": tier},
        "compliance.validate": {"all_required_tools_executed": True,
                                "checks_passed": True},
    }
    tool_calls: list[KYCToolCall] = []
    for i, name in enumerate(_MOCK_TOOLS):
        tool_calls.append(
            KYCToolCall(
                tool_name=name,
                arguments={"case_id": case.case_id},
                result=mock_results[name],
                latency_ms=50 + rng.randint(0, 100),
                started_at=now + timedelta(milliseconds=200 * i),
            )
        )

    reasoning = (
        f"Risk tier {tier}. Customer jurisdiction {case.customer.jurisdiction}; "
        f"sanctions match={sanctions_match}. "
        "All required tools (intake, documents, sanctions, media, risk, decision, "
        "compliance) executed. Decision based on jurisdiction risk tier and "
        "sanctions screening result."
    )

    audit_log_lines = [
        f"[{now.isoformat()}] case_id={case.case_id} tier={tier}",
        f"[{now.isoformat()}] sanctions_match={sanctions_match}",
    ]

    total_latency = sum(t.latency_ms for t in tool_calls)
    return KYCAgentOutput(
        case_id=case.case_id,
        risk_tier=tier,
        sanctions_match=sanctions_match,
        reasoning_chain=reasoning,
        tool_calls=tool_calls,
        audit_log_lines=audit_log_lines,
        latency_ms=total_latency,
        cost_usd=0.02,
        tokens_input=1500,
        tokens_output=400,
    )
