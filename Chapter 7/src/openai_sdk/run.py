"""
run.py
======
Entry point for the OpenAI Agents SDK health-assessment implementation.

Usage:
    python -m src.openai_sdk.run "Assess the financial health of AAPL"

Architecture: orchestrator-worker with synthesis via agents-as-tools
(see the chapter's "Implementation 2: OpenAI Agents SDK" section).
The manager stays in control, invokes both analysts, and writes the
combined assessment.
"""

import asyncio
import sys

from agents import Runner

from .agents import manager_agent


async def run_analysis(query: str) -> str:
    """Run the manager agent, which orchestrates the analysts."""
    result = await Runner.run(manager_agent, query)
    return result.final_output


if __name__ == "__main__":
    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Assess the financial health of AAPL."
    )
    print(f"\nQuery: {query}\n")
    result = asyncio.run(run_analysis(query))
    print(f"\n{'=' * 50}")
    print("ASSESSMENT:")
    print(f"{'=' * 50}")
    print(result)
