"""
common.py
=========
Shared configuration and utility functions for the
Chapter 9 Insurance Workflows.
"""

import os
import uuid
from datetime import date, datetime, timezone
from dotenv import load_dotenv

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_claim_id() -> str:
    """Generate a unique claim identifier."""
    short_id = uuid.uuid4().hex[:8].upper()
    return f"CLM-{date.today().year}-{short_id}"


def log_audit(state: dict, agent_name: str, action: str,
              detail: str = "") -> None:
    """Append an entry to the claim's audit trail."""
    if "claim" not in state:
        return
    claim = state["claim"]
    if "audit_log" not in claim:
        claim["audit_log"] = []
    claim["audit_log"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent_name,
        "action": action,
        "detail": detail[:500],  # Cap detail length
    })


def days_between(d1: str, d2: str) -> int:
    """Signed days from d1 to d2 (positive when d2 is later).

    Accepts ISO dates or datetimes; only the date part is compared.
    """
    fmt = "%Y-%m-%d"
    try:
        return (datetime.strptime(d2[:10], fmt)
                - datetime.strptime(d1[:10], fmt)).days
    except ValueError as exc:
        raise ValueError(
            f"Expected ISO dates (YYYY-MM-DD), got {d1!r}, {d2!r}"
        ) from exc
