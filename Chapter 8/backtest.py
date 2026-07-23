"""
Backtesting the Investment Committee — minimal walk-forward harness.

This is a *framework* backtest, not a research-grade one. It demonstrates:
  - The walk-forward loop structure for evaluating a multi-agent decision system.
  - Equity-curve construction with discrete weekly rebalancing.
  - Standard performance metrics: cumulative return, Sharpe, max drawdown, hit rate.

What it does NOT do, and why:
  - It does NOT use point-in-time fundamentals, news, or macro readings. The
    fetchers in `investment_committee.py` return *current* data, so calling
    `run_committee` on a historical date is technically not a clean backtest.
    For the framework to compute a real out-of-sample Sharpe, every fetcher
    must be refactored to accept an `as_of_date` parameter and return
    historical snapshots from a point-in-time data source.
  - It does NOT account for LLM training-cutoff leakage. If your model saw
    headlines from 2024, its "sentiment" on a 2024 date is partly recall.
    Choose a backtest window that begins strictly AFTER the training cutoff
    of every model in the committee — see the `## Backtesting the committee`
    section in `01_investment_committee.md`.
  - It does NOT model transaction costs, slippage, or borrow costs for shorts.

Run:
    python backtest.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from investment_committee import run_committee


# ---------------------------------------------------------------------------
# Trade record — one entry per (decision_date, ticker) where ACTION != HOLD.
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    decision_date: pd.Timestamp
    ticker: str
    action: str            # "BUY" or "SELL"
    size_pct: float        # PM-proposed sizing as % of NAV
    confidence: int
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float

    @property
    def pnl_pct(self) -> float:
        sign = 1 if self.action == "BUY" else -1
        return sign * (self.exit_price / self.entry_price - 1)

    @property
    def weighted_pnl(self) -> float:
        return (self.size_pct / 100.0) * self.pnl_pct


# ---------------------------------------------------------------------------
# Decision parsing — pull ACTION, CONFIDENCE, SIZE_PCT out of the PM trailer.
# ---------------------------------------------------------------------------
_ACTION_RE = re.compile(r"ACTION\s*=\s*(BUY|SELL|HOLD)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*=\s*(\d+)", re.IGNORECASE)
_SIZE_RE = re.compile(r"SIZE_PCT\s*=\s*([\d.]+)", re.IGNORECASE)


def parse_decision(committee_result: dict) -> tuple[str, int, float]:
    """Extract (ACTION, CONFIDENCE, SIZE_PCT) from the PM thesis, honouring
    the Risk Officer's veto if present.
    """
    thesis = committee_result["pm_thesis"]
    action_match = _ACTION_RE.search(thesis)
    action = action_match.group(1).upper() if action_match else "HOLD"
    conf_match = _CONF_RE.search(thesis)
    size_match = _SIZE_RE.search(thesis)
    confidence = int(conf_match.group(1)) if conf_match else 0
    size_pct = float(size_match.group(1)) if size_match else 0.0

    final = committee_result["final_decision"].lower()
    if "(risk veto)" in final or "(risk rejected)" in final:
        return "HOLD", confidence, 0.0

    return action, confidence, size_pct


# ---------------------------------------------------------------------------
# Walk-forward loop. For each rebalance date, ask the committee about each
# ticker. If it says BUY/SELL, hold the position until the next rebalance.
# ---------------------------------------------------------------------------
def _price_on_or_before(series: pd.Series, ts: pd.Timestamp) -> float | None:
    available = series[series.index <= ts]
    return float(available.iloc[-1]) if len(available) else None


def walk_forward(
    universe: list[str],
    start: str,
    end: str,
    rebalance_freq: str = "W-FRI",     # weekly, Friday close
) -> list[Trade]:
    rebalance_dates = pd.date_range(start, end, freq=rebalance_freq)

    # Pre-fetch all price history once so we can look up entry/exit prices.
    prices = {
        t: yf.Ticker(t).history(start=start, end=end)["Close"]
        for t in universe
    }

    trades: list[Trade] = []
    for i, decision_date in enumerate(rebalance_dates[:-1]):
        exit_date = rebalance_dates[i + 1]
        for ticker in universe:
            # NOTE: run_committee uses LIVE data. See module docstring for
            # the point-in-time refactor that turns this into a real backtest.
            result = run_committee(ticker)
            action, confidence, size_pct = parse_decision(result)
            if action == "HOLD" or size_pct == 0.0:
                continue

            entry = _price_on_or_before(prices[ticker], decision_date)
            exit_ = _price_on_or_before(prices[ticker], exit_date)
            if entry is None or exit_ is None:
                continue

            trades.append(Trade(
                decision_date=decision_date, ticker=ticker,
                action=action, size_pct=size_pct, confidence=confidence,
                entry_price=entry, exit_date=exit_date, exit_price=exit_,
            ))
    return trades


# ---------------------------------------------------------------------------
# Performance metrics.
# ---------------------------------------------------------------------------
def equity_curve(trades: list[Trade], start_nav: float = 1_000_000.0) -> pd.Series:
    """Aggregate per-period weighted PnL into an equity curve indexed by exit_date."""
    if not trades:
        return pd.Series([start_nav])

    df = pd.DataFrame([
        {"exit_date": t.exit_date, "weighted_pnl": t.weighted_pnl}
        for t in trades
    ])
    period_returns = df.groupby("exit_date")["weighted_pnl"].sum()
    return (1 + period_returns).cumprod() * start_nav


def metrics(curve: pd.Series, periods_per_year: float = 52.0) -> dict:
    rets = curve.pct_change().dropna()
    if len(rets) == 0:
        return {"cumulative_return": 0.0, "sharpe": float("nan"),
                "max_drawdown": 0.0, "n_periods": 0}

    cumret = curve.iloc[-1] / curve.iloc[0] - 1
    sharpe = (rets.mean() / rets.std()) * np.sqrt(periods_per_year) \
        if rets.std() > 0 else float("nan")
    running_max = curve.cummax()
    max_dd = (curve / running_max - 1).min()

    return {
        "cumulative_return": float(cumret),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_periods": int(len(rets)),
    }


def hit_rate(trades: list[Trade]) -> float:
    if not trades:
        return float("nan")
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    return wins / len(trades)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    universe = ["TSLA", "MSFT", "NVDA"]
    # Pick a window AFTER your committee's LLM training cutoffs to limit
    # look-ahead leakage. The dates below are illustrative; adjust to a
    # window that starts strictly after the latest model cutoff in use.
    start, end = "2026-01-01", "2026-04-01"

    trades = walk_forward(universe, start, end)
    curve = equity_curve(trades)
    perf = metrics(curve)

    print("=" * 80)
    print(f"Universe         : {', '.join(universe)}")
    print(f"Window           : {start} to {end}  (weekly W-FRI rebalance)")
    print(f"Trades executed  : {len(trades)}")
    print(f"Hit rate         : {hit_rate(trades):.1%}")
    print(f"Cumulative ret.  : {perf['cumulative_return']:.2%}")
    print(f"Sharpe (annl.)   : {perf['sharpe']:.2f}")
    print(f"Max drawdown    : {perf['max_drawdown']:.2%}")
    print("=" * 80)
