"""
agents.py
=========
Agent definitions for the OpenAI Agents SDK health-assessment
implementation.

Architecture: orchestrator-worker with synthesis (see the chapter's
"Implementation 2: OpenAI Agents SDK" section).
The manager invokes specialist agents through agent.as_tool(), which --
unlike handoffs -- keeps the manager in control so it can call several
specialists and synthesize their results into one assessment.
"""

from agents import Agent, function_tool

from .tools import get_profitability_data, get_liquidity_data

MODEL = "gpt-4o"

profitability_analyst = Agent(
    name="Profitability Analyst",
    instructions=(
        "Score the ticker's profitability ratios (ROA, ROE, net "
        "profit margin, gross margin) against the health thresholds "
        "provided by your tool. Give each metric a 1-10 score with a "
        "data-backed justification."
    ),
    tools=[function_tool(get_profitability_data)],
    model=MODEL,
)

liquidity_analyst = Agent(
    name="Liquidity Analyst",
    instructions=(
        "Score the ticker's liquidity and solvency ratios (current "
        "ratio, quick ratio, debt-to-equity, interest coverage) "
        "against the health thresholds provided by your tool. Give "
        "each metric a 1-10 score with a data-backed justification."
    ),
    tools=[function_tool(get_liquidity_data)],
    model=MODEL,
)

manager_agent = Agent(
    name="Health Assessment Manager",
    instructions=(
        "Produce a financial health assessment for the requested "
        "ticker. Call both analyst tools, then synthesize their "
        "scores into an overall verdict with justification. If the "
        "user asks for only one dimension, call only that analyst."
    ),
    tools=[
        profitability_analyst.as_tool(
            tool_name="analyze_profitability",
            tool_description="Score the ticker's profitability "
            "ratios against health thresholds.",
        ),
        liquidity_analyst.as_tool(
            tool_name="analyze_liquidity",
            tool_description="Score the ticker's liquidity and "
            "solvency ratios against health thresholds.",
        ),
    ],
    model=MODEL,
)
