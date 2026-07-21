"""
run.py
======
Entry point for the claims processing pipeline.

Usage:
    python -m src.claims_pipeline.run

Processes a sample auto collision claim through the 7-agent pipeline.
"""

import asyncio
import json

from llama_index.core.agent.workflow import AgentWorkflow

from .agents import (
    fnol_agent, parser_agent, coverage_agent,
    fraud_agent, assessment_agent, decision_agent,
    compliance_agent,
)

# Sample claim data
SAMPLE_CLAIM = {
    "name": "Sarah Chen",
    "policy_number": "POL-2024-AUTO-78432",
    "date_of_loss": "2026-04-08",
    "description": (
        "Rear-ended a stopped delivery truck on I-95 in "
        "heavy rain at approximately 2:47 AM. Front bumper "
        "cracked, headlight broken. Airbags did not deploy. "
        "Experiencing back pain."
    ),
}

SAMPLE_POLICY = {
    "policy_number": "POL-2024-AUTO-78432",
    "policyholder_name": "Sarah Chen",
    "inception_date": "2025-01-15",
    "expiration_date": "2026-07-15",
    "claim_type": "auto",
    "coverage_limit": 50000,
    "deductible": 500,
    "coverage_sections": "collision coverage",
    "exclusions": ["racing", "intentional damage"],
}

SAMPLE_DOCUMENTS = [
    {
        "type": "claim_form",
        "fields": {
            "claimant": "Sarah Chen",
            "date": "2026-04-08",
            "location": "I-95 Northbound, mile marker 42",
            "description": "Rear-end collision with delivery truck",
        },
    },
    {
        "type": "photo",
        "description": (
            "Front bumper cracked with visible impact marks. "
            "Right headlight assembly broken. Paint transfer "
            "from the delivery truck's rear step visible."
        ),
    },
    {
        "type": "photo",
        "description": (
            "Close-up of bumper damage showing crack extending "
            "approximately 18 inches. Bumper partially detached "
            "on right side."
        ),
    },
    {
        "type": "police_report",
        "summary": (
            "Report #2026-04-08-5521. Two-vehicle rear-end "
            "collision. Driver (Chen) struck stopped delivery "
            "vehicle. Weather: rain. Road conditions: wet. "
            "No citations issued. No injuries reported at scene."
        ),
    },
]


async def run_claims_pipeline():
    """Run the full 7-agent claims processing pipeline."""

    workflow = AgentWorkflow(
        agents=[
            fnol_agent, parser_agent, coverage_agent,
            fraud_agent, assessment_agent, decision_agent,
            compliance_agent,
        ],
        root_agent=fnol_agent.name,
        initial_state={
            "claim": {},
            "policy": SAMPLE_POLICY,
            "submitted_documents": SAMPLE_DOCUMENTS,
            "claims_history": [],  # No prior claims
            "fnol_date": "2026-04-08",
        },
    )

    handler = workflow.run(
        user_msg=json.dumps(SAMPLE_CLAIM),
        max_iterations=60,
    )

    # Stream agent transitions (banner only when the agent changes)
    last_agent = None
    async for event in handler.stream_events():
        name = getattr(event, "current_agent_name", None)
        if name and name != last_agent:
            print(f"\n{'=' * 50}")
            print(f"  Agent: {name}")
            print(f"{'=' * 50}")
            last_agent = name
        delta = getattr(event, "delta", None)
        if delta:
            print(delta, end="", flush=True)

    result = await handler

    # Print audit trail from the final workflow state
    print(f"\n\n{'=' * 60}")
    print("  AUDIT TRAIL")
    print(f"{'=' * 60}")
    state = await handler.ctx.store.get("state")
    for entry in state["claim"].get("audit_log", []):
        print(f"  [{entry.get('timestamp', '')}] "
              f"{entry.get('agent', '')}: {entry.get('action', '')}")
    print(result)

    return result


if __name__ == "__main__":
    print("Processing auto collision claim...\n")
    asyncio.run(run_claims_pipeline())
