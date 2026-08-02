"""
tools.py
========
Tool functions for the LlamaIndex multi-agent financial health analyzer.

Each agent has a dedicated tool that reads from and writes to the shared
Context object. This is the shared-state communication pattern described
in the chapter's "Communicating and coordinating between agents" section.
"""

import pandas as pd
from llama_index.core.workflow import Context
from financetoolkit import Toolkit

from ..common import (
    FINANCIAL_MODELING_PREP_API_KEY,
    PROFITABILITY_THRESHOLDS,
    LIQUIDITY_THRESHOLDS,
    format_threshold_prompt,
)


async def get_fundamentals(ctx: Context, ticker: str) -> str:
    """Collect all fundamental ratios for a given ticker.

    Used by: FundamentalAgent (root agent in the pipeline).
    Writes to shared state: ticker, ratios (full DataFrame).
    """
    companies = Toolkit(
        [ticker],
        api_key=FINANCIAL_MODELING_PREP_API_KEY,
        start_date="2022-01-01",
    )
    ratios = companies.ratios.collect_all_ratios()

    current_state = await ctx.store.get("state")
    if current_state["ticker"] == "":
        current_state["ticker"] = ticker

    current_state["ratios"] = ratios
    await ctx.store.set("state", current_state)
    return f"Ratios extracted for {ticker}."


async def get_profitability_ratios(ctx: Context) -> str:
    """Score profitability ratios against health thresholds.

    Used by: ProfitabilityAgent.
    Reads from shared state: ratios, ticker.
    Writes to shared state: profitability_ratios, threshold_profitability_comments.
    """
    current_state = await ctx.store.get("state")
    ratios = current_state["ratios"]
    ticker = current_state["ticker"]

    if current_state["ratios"].empty:
        return "No ratios found. Please run get_fundamentals first."

    # Extract profitability metrics
    ROA = ratios.loc["Return on Assets"]
    ROE = ratios.loc["Return on Equity"]
    net_profit_margin = ratios.loc["Net Profit Margin"]
    gross_margin = ratios.loc["Gross Margin"]

    current_state["profitability_ratios"] = {
        "Return on Assets": ROA,
        "Return on Equity": ROE,
        "Net Profit Margin": net_profit_margin,
        "Gross Margin": gross_margin,
    }

    # Generate threshold prompt for LLM scoring
    threshold_prompt = format_threshold_prompt(
        PROFITABILITY_THRESHOLDS, "Profitability"
    )
    print(f"Profitability ratios extracted for {ticker}.")
    print(f"ROA: {ROA.iloc[-1]:.4f}, ROE: {ROE.iloc[-1]:.4f}")

    current_state["threshold_profitability_comments"] = threshold_prompt
    await ctx.store.set("state", current_state)
    return f"Profitability ratios scored for {ticker}."


async def get_liquidity_ratios(ctx: Context) -> str:
    """Score liquidity ratios against health thresholds.

    Used by: LiquidityAgent.
    Reads from shared state: ratios, ticker.
    Writes to shared state: liquidity_ratios, threshold_liquidity_comments.
    """
    current_state = await ctx.store.get("state")
    ratios = current_state["ratios"]
    ticker = current_state["ticker"]

    if current_state["ratios"].empty:
        return "No ratios found. Please run get_fundamentals first."

    # Extract liquidity metrics
    current_ratio = ratios.loc["Current Ratio"]
    quick_ratio = ratios.loc["Quick Ratio"]
    debt_to_equity = ratios.loc["Debt-to-Equity Ratio"]
    interest_coverage = ratios.loc["Interest Coverage Ratio"]

    current_state["liquidity_ratios"] = {
        "Current Ratio": current_ratio,
        "Quick Ratio": quick_ratio,
        "Debt-to-Equity Ratio": debt_to_equity,
        "Interest Coverage Ratio": interest_coverage,
    }

    # Generate threshold prompt for LLM scoring
    threshold_prompt = format_threshold_prompt(
        LIQUIDITY_THRESHOLDS, "Liquidity"
    )
    print(f"Liquidity ratios extracted for {ticker}.")
    print(f"Current Ratio: {current_ratio.iloc[-1]:.4f}")

    current_state["threshold_liquidity_comments"] = threshold_prompt
    await ctx.store.set("state", current_state)
    return f"Liquidity ratios scored for {ticker}."


async def get_overall_comments(ctx: Context) -> str:
    """Synthesize overall health assessment from individual scores.

    Used by: SupervisorAgent (final agent in the pipeline).
    Reads from shared state: all profitability and liquidity data.
    Writes to shared state: overall_comments.
    """
    current_state = await ctx.store.get("state")

    if current_state["threshold_profitability_comments"] is None:
        return "No profitability comments found."
    if current_state["threshold_liquidity_comments"] is None:
        return "No liquidity comments found."

    current_state["overall_comments"] = "Overall comments done."
    await ctx.store.set("state", current_state)
    return "Overall comments done."
