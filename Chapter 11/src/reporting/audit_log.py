"""Append-only audit logging for evaluation runs.

Every evaluation run is itself a regulatory artifact. The log captures
agent version, dataset version, methodology version, executor, per-case
results, and the aggregate scorecard. Retention follows the firm's
regulatory period — typically 7 years for US financial services.

This implementation writes JSONL to a local file. Production deployments
should write to an immutable store (S3 Object Lock, append-only DB).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..models import Scorecard


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_scorecard(self, scorecard: Scorecard, executor: str) -> None:
        entry = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "executor": executor,
            "agent_version": scorecard.agent_version,
            "dataset_version": scorecard.dataset_version,
            "methodology_version": scorecard.methodology_version,
            "ship_decision": scorecard.ship_decision,
            "weighted_score": scorecard.weighted_score,
            "catastrophic_passes": scorecard.catastrophic_passes,
            "metrics": [m.model_dump() for m in scorecard.metrics],
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
