"""
run.py
======
Entry point for the LlamaIndex multi-agent financial health analyzer.

Usage:
    python -m src.llamaindex.run AAPL
    python -m src.llamaindex.run MSFT

Architecture: sequential pipeline with explicit handoffs (see the
chapter's "Architectural styles for multi-agent financial systems"
section).
"""

import asyncio
import sys

import pandas as pd
from llama_index.core.agent.workflow import (
    AgentStream,
    AgentWorkflow,
    ToolCallResult,
)

from .agents import (
    fundamental_agent,
    profitability_agent,
    liquidity_agent,
    supervisor_agent,
)


async def run_analysis(ticker: str) -> str:
    """Run the full 4-agent financial health analysis pipeline."""

    # Create the workflow with shared initial state
    agent_workflow = AgentWorkflow(
        agents=[
            fundamental_agent,
            profitability_agent,
            liquidity_agent,
            supervisor_agent,
        ],
        root_agent=fundamental_agent.name,
        initial_state={
            "ratios": pd.DataFrame(),
            "ticker": "",
            "profitability_ratios": {},
            "liquidity_ratios": {},
            "threshold_profitability_comments": None,
            "threshold_liquidity_comments": None,
            "overall_comments": None,
        },
    )

    # Run the workflow and stream agent transitions. The default budget
    # of 20 workflow iterations is too tight for a four-agent pipeline
    # (each agent spends roughly three iterations on its tool call,
    # handoff, and response, and any retry burns more).
    handler = agent_workflow.run(
        user_msg=(
            f"Provide the fundamental analysis of {ticker} and "
            f"comments on the financial health of the company."
        ),
        max_iterations=60,
    )

    # Stream events to observe the pipeline: every completed tool call
    # (including the handoffs) prints as it happens, and any text an
    # agent streams appears under that agent's banner. Tool-call-only
    # turns emit empty stream chunks, so the banner is keyed on
    # non-empty deltas.
    last_agent = None
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(
                f"\n[{event.tool_name}] -> "
                f"{str(event.tool_output)[:90]}"
            )
        elif isinstance(event, AgentStream) and event.delta:
            if event.current_agent_name != last_agent:
                print(f"\n{'=' * 50}")
                print(f"  Agent: {event.current_agent_name}")
                print(f"{'=' * 50}")
                last_agent = event.current_agent_name
            print(event.delta, end="", flush=True)

    # Get final result
    result = await handler
    return str(result)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\nAnalyzing {ticker}...\n")
    result = asyncio.run(run_analysis(ticker))
    print(f"\n\n{'=' * 50}")
    print("FINAL RESULT:")
    print(f"{'=' * 50}")
    print(result)
