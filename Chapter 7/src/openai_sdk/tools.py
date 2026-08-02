"""
tools.py
========
Tool functions for the OpenAI Agents SDK health-assessment implementation.

Each analyst agent gets one data tool. The tools do the deterministic
work -- fetching ratios and pairing them with the health-threshold
rubric -- and the agents' LLMs do the judgment (scoring and commentary),
mirroring the division of labor in the LlamaIndex implementation.
"""

import requests

from ..common import (
    FINANCIAL_MODELING_PREP_API_KEY,
    PROFITABILITY_THRESHOLDS,
    LIQUIDITY_THRESHOLDS,
    format_threshold_prompt,
)

# FMP's legacy /api/v3 endpoints are restricted to accounts created
# before August 2025; the /stable API is the current surface. Note the
# field renames: debtEquityRatio -> debtToEquityRatio, and ROA/ROE now
# live in /stable/key-metrics rather than /stable/ratios.
_STABLE_URL = (
    "https://financialmodelingprep.com/stable/{endpoint}"
    "?symbol={symbol}&period=annual&limit=1&apikey={key}"
)

_PROFITABILITY_FIELDS = {
    "Return on Assets": "returnOnAssets",          # key-metrics
    "Return on Equity": "returnOnEquity",          # key-metrics
    "Net Profit Margin": "netProfitMargin",
    "Gross Margin": "grossProfitMargin",
}

_LIQUIDITY_FIELDS = {
    "Current Ratio": "currentRatio",
    "Quick Ratio": "quickRatio",
    "Debt-to-Equity Ratio": "debtToEquityRatio",
    "Interest Coverage Ratio": "interestCoverage",
}


def _fetch(endpoint: str, symbol: str) -> dict:
    url = _STABLE_URL.format(
        endpoint=endpoint, symbol=symbol,
        key=FINANCIAL_MODELING_PREP_API_KEY)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("Error Message"):
        raise RuntimeError(
            f"FMP error for {symbol}: {data['Error Message']}")
    if not isinstance(data, list) or not data:
        return {}
    return data[0]


def _fetch_ratios(symbol: str) -> dict:
    """Merge /stable/ratios with the ROA/ROE from /stable/key-metrics."""
    merged = _fetch("ratios", symbol)
    if merged:
        merged.update({
            k: v for k, v in _fetch("key-metrics", symbol).items()
            if k in ("returnOnAssets", "returnOnEquity")
        })
    return merged


def get_profitability_data(symbol: str) -> str:
    """Fetch the ticker's profitability ratios together with the
    health-threshold rubric the analyst scores them against."""
    raw = _fetch_ratios(symbol)
    if not raw:
        return f"No ratio data available for {symbol}."
    lines = [f"Profitability ratios for {symbol} (latest annual):"]
    for name, field in _PROFITABILITY_FIELDS.items():
        value = raw.get(field)
        lines.append(f"  {name}: {value if value is not None else 'n/a'}")
    lines.append("")
    lines.append(
        format_threshold_prompt(PROFITABILITY_THRESHOLDS, "Profitability"))
    return "\n".join(lines)


def get_liquidity_data(symbol: str) -> str:
    """Fetch the ticker's liquidity and solvency ratios together with
    the health-threshold rubric the analyst scores them against."""
    raw = _fetch_ratios(symbol)
    if not raw:
        return f"No ratio data available for {symbol}."
    lines = [f"Liquidity ratios for {symbol} (latest annual):"]
    for name, field in _LIQUIDITY_FIELDS.items():
        value = raw.get(field)
        lines.append(f"  {name}: {value if value is not None else 'n/a'}")
    lines.append("")
    lines.append(
        format_threshold_prompt(LIQUIDITY_THRESHOLDS, "Liquidity"))
    return "\n".join(lines)
