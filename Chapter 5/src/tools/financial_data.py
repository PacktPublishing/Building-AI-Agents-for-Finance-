"""
tools/financial_data.py
========================
Fetches structured financial metrics from the Financial Datasets API.

Returns normalized CompanyMetrics with consistent units:
- Revenue in millions USD
- Market cap in billions USD
- Margins as percentages (0-100)
- Growth rates as percentages

API docs: https://docs.financialdatasets.ai/
"""

import httpx
from common import CompanyMetrics, FINANCIAL_DATASETS_API_KEY

API_BASE = "https://api.financialdatasets.ai"


async def get_financial_metrics(ticker: str) -> CompanyMetrics:
    """Fetch key financial metrics for a company.

    Retrieves the most recent annual income statement and current price,
    then computes derived metrics (P/E ratio, revenue growth, margins).

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA', 'AMD', 'INTC').

    Returns:
        CompanyMetrics with normalized financial data.

    Raises:
        ValueError: If ticker not found or API returns an error.
    """
    headers = {"X-API-Key": FINANCIAL_DATASETS_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch income statements (last 2 years for growth calculation)
        income_resp = await client.get(
            f"{API_BASE}/financials/income-statements",
            params={"ticker": ticker, "period": "annual", "limit": 2},
            headers=headers,
        )
        income_resp.raise_for_status()
        income_data = income_resp.json().get("income_statements", [])

        if not income_data:
            raise ValueError(f"No income statement data found for {ticker}")

        # Fetch current price snapshot
        price_resp = await client.get(
            f"{API_BASE}/prices/snapshot",
            params={"ticker": ticker},
            headers=headers,
        )
        price_resp.raise_for_status()
        snapshot = price_resp.json().get("snapshot", {})

    current = income_data[0]
    previous = income_data[1] if len(income_data) > 1 else None

    # Extract and normalize values
    revenue_raw = current.get("revenue", 0) or 0
    revenue = revenue_raw / 1_000_000  # Convert to millions

    eps = current.get("eps_diluted") or current.get("eps_basic", 0) or 0
    price = snapshot.get("price", 0) or 0

    # Compute derived metrics
    revenue_growth = 0.0
    if previous:
        prev_revenue = previous.get("revenue", 0) or 0
        if prev_revenue > 0:
            revenue_growth = ((revenue_raw - prev_revenue) / prev_revenue) * 100

    pe_ratio = 0.0
    if eps > 0 and price > 0:
        pe_ratio = price / eps

    gross_margin = (current.get("gross_margin", 0) or 0) * 100
    operating_margin = (current.get("operating_margin", 0) or 0) * 100
    market_cap = (snapshot.get("market_cap", 0) or 0) / 1_000_000_000

    company_name = current.get("company_name", ticker)
    period = current.get("period", "FY2024")

    return CompanyMetrics(
        ticker=ticker.upper(),
        company_name=company_name,
        revenue=round(revenue, 1),
        revenue_growth=round(revenue_growth, 1),
        eps=round(eps, 2),
        pe_ratio=round(pe_ratio, 1),
        gross_margin=round(gross_margin, 1),
        operating_margin=round(operating_margin, 1),
        market_cap=round(market_cap, 1),
        period=period,
    )


async def get_price_history(
    ticker: str, days: int = 30
) -> list[dict]:
    """Fetch recent price history for a stock.

    Args:
        ticker: Stock ticker symbol.
        days: Number of days of history to fetch.

    Returns:
        List of dicts with 'date', 'open', 'high', 'low', 'close', 'volume'.
    """
    headers = {"X-API-Key": FINANCIAL_DATASETS_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{API_BASE}/prices",
            params={"ticker": ticker, "interval": "day", "limit": days},
            headers=headers,
        )
        resp.raise_for_status()

    return resp.json().get("prices", [])
