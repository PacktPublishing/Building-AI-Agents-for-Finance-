"""
common.py
=========
Shared tooling for the Chapter 3 framework bake-off.

Every implementation runs the same task against the same deterministic
mock data, so differences in the outputs reflect the frameworks — not
the inputs. Swapping in a live market-data source (yfinance, or your
vendor feed) is a one-line change in MOCK_STOCK_DATA / get_stock_data_tool_func.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Deterministic mock data. Live alternative:
#   import yfinance as yf
#   t = yf.Ticker(ticker); price = t.fast_info["lastPrice"]; ...
MOCK_STOCK_DATA = {
    "AAPL": {"price": 195.30, "eps": 6.67},
    "JPM": {"price": 148.70, "eps": 12.61},
}

# One small model for every OpenAI-backed implementation so the
# comparison stays fair. Update to the current small tier before running.
OPENAI_MODEL = os.getenv("BAKEOFF_OPENAI_MODEL", "gpt-5-mini")

system_message = (
    "You are a precise financial analyst. Use the provided tools to "
    "fetch stock data and compute P/E ratios — never invent numbers. "
    "Finish with a short, professional investment memo."
)

input_message = (
    "Compare Apple (AAPL) and JPMorgan (JPM) on P/E ratios and "
    "summarize in a memo."
)


@dataclass
class StockData:
    ticker: str
    price: float
    eps: float


def check_api_key(name: str = "OPENAI_API_KEY") -> None:
    if not os.getenv(name):
        raise SystemExit(
            f"{name} is not set. Export it (or add it to .env) first.")


def get_stock_data_tool_func(ticker: str) -> StockData:
    """Return the latest price and EPS for a ticker (mock data)."""
    row = MOCK_STOCK_DATA.get(ticker.upper())
    if row is None:
        raise ValueError(f"No data for ticker {ticker!r}")
    return StockData(ticker=ticker.upper(), price=row["price"], eps=row["eps"])


def compute_pe_tool_func(price: float, eps: float) -> float:
    """Compute the price-to-earnings ratio."""
    if eps == 0:
        raise ValueError("EPS is zero; P/E undefined")
    return round(price / eps, 2)


def persist_framework_output(framework: str, output: str,
                             token_usage: str = "") -> Path:
    """Write a framework's run output to outputs/<framework>.txt."""
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{framework}.txt"
    header = f"# {framework} — bake-off output\n"
    if token_usage:
        header += f"# token usage: {token_usage}\n"
    path.write_text(header + "\n" + output.strip() + "\n")
    print(f"[saved] {path}")
    return path
