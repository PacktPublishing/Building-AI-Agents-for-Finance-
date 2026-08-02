"""
parallel_analysis.py
====================
Lab extension ("Extension: Parallel multi-company analysis"): runs the
4-agent financial health pipeline for multiple companies concurrently
using asyncio.gather().

Demonstrates the Parallelization architectural style (see the chapter's
"Architectural styles for multi-agent financial systems" section):
independent tasks distributed across concurrent pipeline instances.

Usage:
    python -m src.extensions.parallel_analysis
"""

import asyncio
import time
import sys

from ..llamaindex.run import run_analysis


async def parallel_portfolio_analysis(
    tickers: list[str],
) -> list[dict]:
    """Run financial health analysis for multiple companies in parallel.

    Each company gets its own independent 4-agent pipeline instance.
    All instances run concurrently via asyncio.gather().
    """
    print(f"Analyzing {len(tickers)} companies in parallel...")
    print(f"Tickers: {', '.join(tickers)}\n")

    start_time = time.time()

    # Launch all analyses concurrently
    tasks = [run_analysis(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time

    # Process results
    analyses = []
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            analyses.append({
                "ticker": ticker,
                "status": "failed",
                "error": str(result),
            })
        else:
            analyses.append({
                "ticker": ticker,
                "status": "success",
                "assessment": result,
            })

    print(f"\n{'=' * 60}")
    print(f"  Parallel analysis complete: {len(tickers)} companies")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Average per company: {elapsed / len(tickers):.1f}s")
    print(f"  Sequential estimate (assumes ~30s/company): ~{30 * len(tickers)}s")
    print(f"  Estimated speedup vs that assumption: "
          f"~{(30 * len(tickers)) / elapsed:.1f}x")
    print(f"{'=' * 60}\n")

    return analyses


async def sequential_portfolio_analysis(
    tickers: list[str],
) -> list[dict]:
    """Run analyses sequentially for comparison with parallel version."""
    print(f"Analyzing {len(tickers)} companies sequentially...")

    start_time = time.time()

    analyses = []
    for ticker in tickers:
        result = await run_analysis(ticker)
        analyses.append({
            "ticker": ticker,
            "status": "success",
            "assessment": result,
        })

    elapsed = time.time() - start_time
    print(f"Sequential time: {elapsed:.1f}s")

    return analyses


if __name__ == "__main__":
    # Default portfolio: 5 major companies across sectors
    tickers = sys.argv[1:] if len(sys.argv) > 1 else [
        "AAPL",   # Technology
        "MSFT",   # Technology
        "GOOGL",  # Technology
        "NVDA",   # Semiconductors
        "JPM",    # Financial Services
    ]

    results = asyncio.run(parallel_portfolio_analysis(tickers))

    # Print summary table
    print(f"\n{'Ticker':<8} {'Status':<10} {'Assessment (first 100 chars)'}")
    print("-" * 60)
    for r in results:
        summary = (
            r.get("assessment", r.get("error", ""))[:100]
            if r["status"] == "success"
            else r.get("error", "")[:100]
        )
        print(f"{r['ticker']:<8} {r['status']:<10} {summary}")
