"""
tools/
======
Financial data tools for the Deep Search Agent.

Each module wraps a specific data source and returns clean,
normalized data that the agent can directly reason about.
"""

from tools.financial_data import get_financial_metrics
from tools.sec_filings import get_risk_factors
from tools.web_search import search_financial_news

__all__ = [
    "get_financial_metrics",
    "get_risk_factors",
    "search_financial_news",
]
