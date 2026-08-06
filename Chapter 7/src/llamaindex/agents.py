"""
agents.py
=========
Agent definitions for the LlamaIndex multi-agent financial health analyzer.

Each agent has:
  - A focused system prompt defining its role
  - A dedicated tool function
  - An explicit can_handoff_to declaring valid transitions

Architecture: sequential pipeline (see the chapter's "Architectural
styles for multi-agent financial systems" section).
Handoff pattern: explicit next-agent declaration (see "Communicating
and coordinating between agents").

Two disciplines keep runs deterministic: every prompt tells its agent
to call its tool exactly once before handing off (a transient tool
failure must not become a retry loop), and the SupervisorAgent -- the
last agent -- writes the final answer itself instead of handing off.
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from .tools import (
    get_fundamentals,
    get_profitability_ratios,
    get_liquidity_ratios,
    get_overall_comments,
)

# Use a capable model for analysis and synthesis
llm = OpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# Agent 1: FundamentalAgent (root agent)
# ---------------------------------------------------------------------------
fundamental_agent = FunctionAgent(
    name="FundamentalAgent",
    description="Collect fundamental ratios for a given ticker.",
    system_prompt=(
        "You are the fundamental analyst that can extract different "
        "fundamental ratios for a given ticker.\n"
        "Call get_fundamentals exactly once for the requested ticker; "
        "never call it a second time.\n"
        "Once you have extracted the fundamental financial ratios, "
        "you should hand off control to the ProfitabilityAgent to "
        "extract the profitability ratios.\n"
    ),
    llm=llm,
    tools=[get_fundamentals],
    can_handoff_to=["ProfitabilityAgent"],
)


# ---------------------------------------------------------------------------
# Agent 2: ProfitabilityAgent
# ---------------------------------------------------------------------------
profitability_agent = FunctionAgent(
    name="ProfitabilityAgent",
    description=(
        "Collect profitability ratios for a given ticker: ROA, ROE, "
        "Net Profit Margin, and Gross Margin, and score these ratios "
        "from 1 to 10 against a set of threshold values."
    ),
    system_prompt=(
        "You are the ProfitabilityAgent that can collect profitability "
        "ratios. You collect these ratios from the FundamentalAgent.\n"
        "Call get_profitability_ratios exactly once.\n"
        "Once these ratios are collected as profitability_ratios, you should "
        "score each ratio from 1 to 10 against the threshold values "
        "provided in threshold_profitability_comments, with a data-backed "
        "justification for each. DO NOT ADD anything else.\n"
        "Once the comments are done, you should hand off control to the "
        "LiquidityAgent.\n"
    ),
    llm=llm,
    tools=[get_profitability_ratios],
    can_handoff_to=["LiquidityAgent"],
)


# ---------------------------------------------------------------------------
# Agent 3: LiquidityAgent
# ---------------------------------------------------------------------------
liquidity_agent = FunctionAgent(
    name="LiquidityAgent",
    description=(
        "Collect liquidity ratios for a given ticker: Current Ratio, "
        "Quick Ratio, Debt-to-Equity Ratio, Interest Coverage Ratio, "
        "and score these ratios from 1 to 10 against a set of threshold values."
    ),
    system_prompt=(
        "You are the LiquidityAgent that can collect liquidity ratios.\n"
        "You collect these ratios from the FundamentalAgent.\n"
        "Call get_liquidity_ratios exactly once.\n"
        "Once these ratios are collected as liquidity_ratios, you should "
        "score each ratio from 1 to 10 against the threshold values "
        "provided in threshold_liquidity_comments, with a data-backed "
        "justification for each. DO NOT ADD anything else.\n"
        "Once the comments are done, you should hand off control to the "
        "SupervisorAgent.\n"
    ),
    llm=llm,
    tools=[get_liquidity_ratios],
    can_handoff_to=["SupervisorAgent"],
)


# ---------------------------------------------------------------------------
# Agent 4: SupervisorAgent
# ---------------------------------------------------------------------------
supervisor_agent = FunctionAgent(
    name="SupervisorAgent",
    description=(
        "Provide a final comment on the overall health of the firm "
        "based on the different comments coming from the different agents."
    ),
    system_prompt=(
        "You are a fundamental analyst and supervisor. You collect "
        "comments coming from various agents such as ProfitabilityAgent "
        "and LiquidityAgent.\n"
        "Call get_overall_comments exactly once. Then write the final "
        "health report yourself: list every profitability and liquidity "
        "metric with its 1-10 score and a one-line justification, then "
        "give an overall verdict on the company.\n"
        "Justify your verdict with details and data.\n"
        "You are the last agent in the pipeline: write the final answer "
        "yourself and do not hand off.\n"
    ),
    llm=llm,
    tools=[get_overall_comments],
    # An empty list makes this agent terminal. Omitting the argument
    # would mean "can hand off to ANY agent" in AgentWorkflow.
    can_handoff_to=[],
)
