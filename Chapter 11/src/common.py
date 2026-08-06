"""Shared configuration, paths, and small utilities.

Keep this file tiny. Anything that grows belongs in its own module.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "kyc_eval"
SAMPLE_REPORTS_DIR = ROOT / "data" / "sample_reports"

DEFAULT_AGENT_MODEL = os.environ.get("KYC_AGENT_MODEL", "claude-sonnet-4-6")
DEFAULT_JUDGE_MODEL = os.environ.get("KYC_JUDGE_MODEL", "claude-opus-5")

LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET = os.environ.get("LANGFUSE_SECRET_KEY")

AGENT_VERSION = os.environ.get("KYC_AGENT_VERSION", "dev")
DATASET_VERSION = "v1.0.0"
METHODOLOGY_VERSION = "v1.0.0"


def require_env(*names: str) -> None:
    """Raise if any required env var is unset.

    Used at the top of CLI entry points so the failure mode is clear, not
    a confusing API error 200 calls in.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(f"missing env vars: {', '.join(missing)}")
