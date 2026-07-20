"""
agents.py
=========
Agent definitions for the claims processing pipeline.

Architecture: Sequential pipeline with guardrails (Ch 7, Section 2.1).
Each agent has a focused system prompt, a dedicated tool, and an
explicit can_handoff_to declaring the next agent in the sequence.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from .tools import (
    process_fnol,
    parse_documents,
    verify_coverage,
    screen_for_fraud,
    assess_damage,
    make_decision,
    validate_compliance,
)

llm = OpenAI(model="gpt-4o", temperature=0)

# --- Agent 1: FNOL Agent (root) ---
fnol_agent = FunctionAgent(
    name="FNOLAgent",
    description="Process first notice of loss.",
    system_prompt=(
        "You handle initial claim intake. Classify the "
        "claim type, create the claim record, then hand "
        "off to the DocumentParser."
    ),
    llm=llm,
    tools=[process_fnol],
    can_handoff_to=["DocumentParser"],
)

# --- Agent 2: Document Parser ---
parser_agent = FunctionAgent(
    name="DocumentParser",
    description="Parse and extract data from documents.",
    system_prompt=(
        "FIRST call the parse_documents tool to extract "
        "structured data from every submitted document. "
        "Then IMMEDIATELY hand off to CoverageAgent. "
        "Never answer the user directly."
    ),
    llm=llm,
    tools=[parse_documents],
    can_handoff_to=["CoverageAgent"],
)

# --- Agent 3: Coverage Verification ---
coverage_agent = FunctionAgent(
    name="CoverageAgent",
    description="Verify policy coverage for the claim.",
    system_prompt=(
        "You verify whether the claimed event is covered "
        "by the policy. Check limits, deductibles, and "
        "exclusions. Hand off to FraudScreener."
    ),
    llm=llm,
    tools=[verify_coverage],
    can_handoff_to=["FraudScreener"],
)

# --- Agent 4: Fraud Screening ---
fraud_agent = FunctionAgent(
    name="FraudScreener",
    description="Screen claim for fraud indicators.",
    system_prompt=(
        "You screen claims against fraud indicators. "
        "Check red flags, compute a risk score, and "
        "recommend proceed or investigate. "
        "Hand off to AssessmentAgent."
    ),
    llm=llm,
    tools=[screen_for_fraud],
    can_handoff_to=["AssessmentAgent"],
)

# --- Agent 5: Damage Assessment ---
assessment_agent = FunctionAgent(
    name="AssessmentAgent",
    description="Estimate the claim payout amount.",
    system_prompt=(
        "FIRST call the assess_damage tool to compute the "
        "payout (it applies the deductible and coverage "
        "limit). Then IMMEDIATELY hand off to DecisionAgent. "
        "Never answer the user directly."
    ),
    llm=llm,
    tools=[assess_damage],
    can_handoff_to=["DecisionAgent"],
)

# --- Agent 6: Decision ---
decision_agent = FunctionAgent(
    name="DecisionAgent",
    description="Make the final claim decision.",
    system_prompt=(
        "FIRST call the make_decision tool to record the "
        "decision (APPROVE, DENY, or ESCALATE) with its "
        "reasoning chain. Then IMMEDIATELY hand off to "
        "ComplianceAgent. Never answer the user directly."
    ),
    llm=llm,
    tools=[make_decision],
    can_handoff_to=["ComplianceAgent"],
)

# --- Agent 7: Compliance (Guardrail) ---
compliance_agent = FunctionAgent(
    name="ComplianceAgent",
    description="Validate decision against regulations.",
    system_prompt=(
        "You are the compliance guardrail. Validate the "
        "decision against regulatory requirements. Check "
        "denial reason validity, reasoning completeness, "
        "and pipeline stage completion. You have VETO "
        "power — block any non-compliant decision."
    ),
    llm=llm,
    tools=[validate_compliance],
    can_handoff_to=[],  # terminal agent — pipeline ends here
)
