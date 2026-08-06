"""
common.py
=========
Shared configuration, financial health thresholds, and utility functions
for the Chapter 7 Multi-Agent Financial Health Analyzer.

The LlamaIndex and OpenAI Agents SDK implementations share these
definitions to ensure consistent behavior; the AutoGen version
coordinates through generated code and does not use this module.
"""

import os
from dotenv import load_dotenv

_ = load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINANCIAL_MODELING_PREP_API_KEY = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# Financial Health Thresholds
# ---------------------------------------------------------------------------
# These thresholds come from standard financial analysis practice.
# Each metric is scored 1-10 based on where the value falls.

PROFITABILITY_THRESHOLDS = {
    "Return on Assets": {
        "healthy": 0.05,      # > 5%
        "moderate": 0.02,     # 2% - 5%
        "description": "Measures how efficiently assets generate profit",
    },
    "Return on Equity": {
        "healthy": 0.15,      # > 15%
        "moderate": 0.08,     # 8% - 15%
        "description": "Measures return generated on shareholder equity",
    },
    "Net Profit Margin": {
        "healthy": 0.10,      # > 10%
        "moderate": 0.05,     # 5% - 10%
        "description": "Percentage of revenue retained as profit",
    },
    "Gross Margin": {
        "healthy": 0.40,      # > 40%
        "moderate": 0.20,     # 20% - 40%
        "description": "Percentage of revenue after cost of goods sold",
    },
}

LIQUIDITY_THRESHOLDS = {
    "Current Ratio": {
        "healthy_low": 1.5,
        "healthy_high": 3.0,
        "warning": 1.0,       # Below 1.0 is a warning
        "description": "Ability to pay short-term obligations",
    },
    "Quick Ratio": {
        "healthy": 1.0,       # > 1.0
        "description": "Liquid assets vs. short-term liabilities",
    },
    "Debt-to-Equity Ratio": {
        "healthy_low": 0.3,
        "healthy_high": 1.5,
        "warning": 2.0,       # Above 2.0 is high leverage
        "description": "Balance between debt and equity financing",
    },
    "Interest Coverage Ratio": {
        "healthy": 3.0,       # > 3.0
        "moderate": 1.5,      # 1.5 - 3.0
        "description": "Ability to service debt interest payments",
    },
}


def format_threshold_prompt(thresholds: dict, category: str) -> str:
    """Format thresholds into a prompt string for the LLM."""
    lines = [f"## {category} Thresholds\n"]
    for metric, config in thresholds.items():
        lines.append(f"### {metric}")
        lines.append(f"Description: {config['description']}")
        as_pct = category.lower().startswith("profit")
        for key, value in config.items():
            if key != "description":
                if isinstance(value, float) and as_pct:
                    lines.append(f"  {key}: {value:.0%}")
                else:
                    # Liquidity metrics are plain ratios (1.5, 3.0),
                    # not percentages -- do not render 1.5 as 150%.
                    lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines)
