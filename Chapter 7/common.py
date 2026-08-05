"""
common.py — Shared configuration and helpers for Chapter 7 multi-agent labs.

All three framework implementations (LlamaIndex, OpenAI Agents SDK, AutoGen)
share these definitions to keep the *task* identical across labs and let you
focus on the *architecture* each framework expresses.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Financial health thresholds — used by the Profitability and Liquidity agents
# to score companies on a 1-10 scale.
# ---------------------------------------------------------------------------

PROFITABILITY_THRESHOLDS = {
    "Return on Assets": {
        "healthy": 0.05,
        "moderate": 0.02,
        "description": "How efficiently assets generate profit",
    },
    "Return on Equity": {
        "healthy": 0.15,
        "moderate": 0.08,
        "description": "Return generated on shareholder equity",
    },
    "Net Profit Margin": {
        "healthy": 0.10,
        "moderate": 0.05,
        "description": "Percentage of revenue retained as profit",
    },
    "Gross Margin": {
        "healthy": 0.40,
        "moderate": 0.20,
        "description": "Percentage of revenue after cost of goods sold",
    },
}

LIQUIDITY_THRESHOLDS = {
    "Current Ratio": {
        "healthy_low": 1.5,
        "healthy_high": 3.0,
        "warning": 1.0,
        "description": "Ability to pay short-term obligations",
    },
    "Quick Ratio": {
        "healthy": 1.0,
        "description": "Liquid assets vs short-term liabilities",
    },
    "Debt-to-Equity Ratio": {
        "healthy_low": 0.3,
        "healthy_high": 1.5,
        "warning": 2.0,
        "description": "Balance between debt and equity financing",
    },
    "Interest Coverage Ratio": {
        "healthy": 3.0,
        "moderate": 1.5,
        "description": "Ability to service debt interest payments",
    },
}


def format_threshold_prompt(thresholds: dict, category: str) -> str:
    """Render a threshold dict into a prompt snippet for an LLM agent."""
    lines = [f"## {category} Thresholds\n"]
    as_pct = category.lower().startswith("profit")
    for metric, config in thresholds.items():
        lines.append(f"### {metric}")
        lines.append(f"Description: {config['description']}")
        for key, value in config.items():
            if key == "description":
                continue
            if isinstance(value, float) and as_pct:
                lines.append(f"  {key}: {value:.0%}")
            else:
                # Liquidity metrics are plain ratios (1.5, 3.0), not
                # percentages -- do not render 1.5 as 150%.
                lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stub financial data — keeps labs reproducible without external API keys.
# Real implementations call the Financial Modeling Prep API (Ch 7 source repo).
# ---------------------------------------------------------------------------

STUB_FUNDAMENTALS = {
    "AAPL": {
        "profitability": {
            "Return on Assets": 0.27,
            "Return on Equity": 1.56,
            "Net Profit Margin": 0.25,
            "Gross Margin": 0.46,
        },
        "liquidity": {
            "Current Ratio": 0.99,
            "Quick Ratio": 0.84,
            "Debt-to-Equity Ratio": 1.95,
            "Interest Coverage Ratio": 28.74,
        },
        "stock_price": {
            "symbol": "AAPL", "price": 195.30, "volume": 52_000_000,
            "priceAvg50": 188.10, "priceAvg200": 178.45,
            "eps": 6.67, "pe": 29.3, "earningsAnnouncement": "2026-07-31",
        },
        "company": {
            "companyName": "Apple Inc.", "industry": "Consumer Electronics",
            "sector": "Technology", "mktCap": 3_010_000_000_000,
        },
        "income": {
            "date": "2025-09-30", "revenue": 391_035_000_000,
            "gross_profit": 180_683_000_000, "net_income": 96_995_000_000,
            "ebitda": 134_661_000_000, "EPS": 6.67, "EPS_diluted": 6.61,
        },
    },
    "MSFT": {
        "profitability": {
            "Return on Assets": 0.18,
            "Return on Equity": 0.39,
            "Net Profit Margin": 0.36,
            "Gross Margin": 0.70,
        },
        "liquidity": {
            "Current Ratio": 1.27,
            "Quick Ratio": 1.24,
            "Debt-to-Equity Ratio": 0.40,
            "Interest Coverage Ratio": 38.40,
        },
        "stock_price": {
            "symbol": "MSFT", "price": 415.00, "volume": 21_000_000,
            "priceAvg50": 410.20, "priceAvg200": 388.10,
            "eps": 11.80, "pe": 35.2, "earningsAnnouncement": "2026-07-25",
        },
        "company": {
            "companyName": "Microsoft Corporation", "industry": "Software",
            "sector": "Technology", "mktCap": 3_100_000_000_000,
        },
        "income": {
            "date": "2025-06-30", "revenue": 245_122_000_000,
            "gross_profit": 171_008_000_000, "net_income": 88_136_000_000,
            "ebitda": 130_109_000_000, "EPS": 11.80, "EPS_diluted": 11.75,
        },
    },
    "NVDA": {
        "profitability": {
            "Return on Assets": 0.53,
            "Return on Equity": 1.15,
            "Net Profit Margin": 0.48,
            "Gross Margin": 0.73,
        },
        "liquidity": {
            "Current Ratio": 4.20,
            "Quick Ratio": 3.50,
            "Debt-to-Equity Ratio": 0.40,
            "Interest Coverage Ratio": 142.00,
        },
        "stock_price": {
            "symbol": "NVDA", "price": 938.40, "volume": 38_000_000,
            "priceAvg50": 880.20, "priceAvg200": 790.15,
            "eps": 12.96, "pe": 72.4, "earningsAnnouncement": "2026-08-21",
        },
        "company": {
            "companyName": "NVIDIA Corporation", "industry": "Semiconductors",
            "sector": "Technology", "mktCap": 2_891_000_000_000,
        },
        "income": {
            "date": "2025-01-26", "revenue": 60_922_000_000,
            "gross_profit": 44_301_000_000, "net_income": 29_760_000_000,
            "ebitda": 35_583_000_000, "EPS": 12.96, "EPS_diluted": 12.86,
        },
    },
    "GOOGL": {
        "profitability": {
            "Return on Assets": 0.22,
            "Return on Equity": 0.30,
            "Net Profit Margin": 0.27,
            "Gross Margin": 0.57,
        },
        "liquidity": {
            "Current Ratio": 2.10,
            "Quick Ratio": 2.06,
            "Debt-to-Equity Ratio": 0.10,
            "Interest Coverage Ratio": 87.50,
        },
        "stock_price": {
            "symbol": "GOOGL", "price": 168.40, "volume": 27_000_000,
            "priceAvg50": 162.50, "priceAvg200": 152.30,
            "eps": 7.54, "pe": 22.3, "earningsAnnouncement": "2026-07-23",
        },
        "company": {
            "companyName": "Alphabet Inc.", "industry": "Internet Content & Information",
            "sector": "Communication Services", "mktCap": 2_080_000_000_000,
        },
        "income": {
            "date": "2025-12-31", "revenue": 350_018_000_000,
            "gross_profit": 199_510_000_000, "net_income": 100_118_000_000,
            "ebitda": 134_276_000_000, "EPS": 7.54, "EPS_diluted": 7.40,
        },
    },
    "JPM": {
        "profitability": {
            "Return on Assets": 0.014,
            "Return on Equity": 0.17,
            "Net Profit Margin": 0.36,
            "Gross Margin": 1.00,  # Banks report differently; gross margin is conventionally ~1.
        },
        "liquidity": {
            "Current Ratio": 1.00,  # Banks use different liquidity metrics; this is a placeholder.
            "Quick Ratio":   1.00,
            "Debt-to-Equity Ratio": 1.40,
            "Interest Coverage Ratio": 4.50,
        },
        "stock_price": {
            "symbol": "JPM", "price": 198.70, "volume": 11_000_000,
            "priceAvg50": 194.00, "priceAvg200": 185.50,
            "eps": 16.23, "pe": 12.2, "earningsAnnouncement": "2026-07-15",
        },
        "company": {
            "companyName": "JPMorgan Chase & Co.", "industry": "Banks—Diversified",
            "sector": "Financial Services", "mktCap": 568_000_000_000,
        },
        "income": {
            "date": "2025-12-31", "revenue": 158_104_000_000,
            "gross_profit": 158_104_000_000,  # No COGS for banks.
            "net_income": 49_552_000_000,
            "ebitda": 0,  # Not meaningful for banks.
            "EPS": 16.23, "EPS_diluted": 16.07,
        },
    },
}


def stub_lookup(ticker: str) -> dict:
    """Return the canned data block for a ticker, raising on unknown tickers."""
    t = ticker.upper()
    if t not in STUB_FUNDAMENTALS:
        raise ValueError(
            f"No stub data for {t}. Add it to STUB_FUNDAMENTALS or "
            "wire in the real Financial Modeling Prep API."
        )
    return STUB_FUNDAMENTALS[t]
