"""
Multi-Agent Investment Committee — implemented as a LangGraph state graph.

Roles:
  - Specialist analysts (Fundamentals, Technicals, Sentiment, Macro) run in parallel.
  - The Portfolio Manager synthesises their reports into a thesis.
  - The Risk Officer applies hard rules and may veto.

LLMs:
  - Sonnet 4.6 for specialists (cost-efficient, strong tool use).
  - Haiku 4.5 for high-volume sentiment work.
  - Opus 4.7 for the PM and Risk roles where reasoning quality matters most.

Run:
    export ANTHROPIC_API_KEY=...
    python investment_committee.py
"""
from __future__ import annotations

import re
from typing import TypedDict

import yfinance as yf
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Models — one instance per tier, reused across nodes.
# ---------------------------------------------------------------------------
# We never set `temperature`. Claude models released after Opus 4.6 (Opus 4.7
# and later) no longer accept temperature/top_p/top_k: any value other than 1.0
# is rejected with a 400 error. Sonnet 4.6 / Haiku 4.5 still accept it today, but
# omitting it everywhere keeps this code runnable as those models are retired and
# replaced by newer (temperature-less) ones. Behaviour is steered via prompts.
opus = ChatAnthropic(model="claude-opus-4-7", max_tokens=2000)
sonnet = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1500)
haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=800)


# ---------------------------------------------------------------------------
# Shared state — every node reads from and writes into this TypedDict.
# Each analyst writes a different field, so no reducer is needed for the
# parallel fan-out from START.
# ---------------------------------------------------------------------------
class CommitteeState(TypedDict):
    ticker: str
    fundamentals_report: str
    technicals_report: str
    sentiment_report: str
    macro_report: str
    pm_thesis: str
    risk_verdict: str
    final_decision: str


# ---------------------------------------------------------------------------
# Data fetchers — small, deterministic Python functions that the analyst
# nodes call before invoking the LLM. Keeping the numerics in Python (not in
# the prompt) makes them auditable and avoids hallucinated figures.
# ---------------------------------------------------------------------------
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info or {}
    return {
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "returnOnEquity": info.get("returnOnEquity"),
        "profitMargins": info.get("profitMargins"),
        "debtToEquity": info.get("debtToEquity"),
        "currentRatio": info.get("currentRatio"),
        "revenueGrowth": info.get("revenueGrowth"),
        "earningsGrowth": info.get("earningsGrowth"),
    }


def compute_indicators(ticker: str, period: str = "1y") -> dict:
    df = yf.Ticker(ticker).history(period=period)
    close = df["Close"]
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + gain / loss)).iloc[-1]
    vol_30d = close.pct_change().rolling(30).std().iloc[-1] * (252 ** 0.5)
    return {
        "last_close": float(close.iloc[-1]),
        "sma_50": float(sma_50) if sma_50 is not None else None,
        "sma_200": float(sma_200) if sma_200 is not None else None,
        "rsi_14": float(rsi),
        "vol_30d_annualized": float(vol_30d),
    }


def _is_about_company(title: str, ticker: str, short_name: str) -> bool:
    """True if the article title names the ticker or a meaningful word from
    the company's short name. We require a *title* mention because yfinance's
    per-ticker news feed mixes in market-wide articles where the company is
    only a passing reference. Word boundaries on the ticker avoid false
    positives for short symbols (e.g., "T" for AT&T matching every "the").
    """
    title_text = title.lower()
    if re.search(rf"\b{re.escape(ticker.lower())}\b", title_text):
        return True
    for word in short_name.lower().split():
        word = word.rstrip(".,:")
        if len(word) >= 4 and word in title_text:
            return True
    return False


def fetch_news(ticker: str, n: int = 8) -> list[str]:
    """Fetch up to n recent headlines that are *primarily* about `ticker`.

    yfinance's per-ticker news feed sometimes returns market-wide articles
    that only tangentially mention the company; we filter those out. Each
    returned line has the format '[date, publisher] title — first sentence'
    so the sentiment analyst gets date and publisher context, not just a bare
    headline.
    """
    try:
        stock = yf.Ticker(ticker)
        short_name = (stock.info or {}).get("shortName", "")
        items = stock.news or []
    except Exception:
        items, short_name = [], ""

    headlines: list[str] = []
    for it in items:
        content = it.get("content", it)
        title = content.get("title") or it.get("title")
        if not title or not _is_about_company(title, ticker, short_name):
            continue

        publisher = (content.get("provider") or {}).get("displayName", "")
        pub_date = content.get("pubDate") or ""
        if isinstance(pub_date, str) and "T" in pub_date:
            pub_date = pub_date.split("T", 1)[0]

        summary = content.get("summary") or content.get("description") or ""
        first_sentence = ""
        if summary:
            m = re.search(r"([^.!?]*[.!?])", summary.strip())
            first_sentence = m.group(1).strip() if m else summary.strip()[:200]

        meta_parts = [p for p in (pub_date, publisher) if p]
        meta = f"[{', '.join(meta_parts)}] " if meta_parts else ""
        line = f"{meta}{title}"
        if first_sentence:
            line += f" — {first_sentence}"
        headlines.append(line)

        if len(headlines) >= n:
            break

    return headlines


def fetch_macro_snapshot() -> dict:
    vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
    tnx = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
    spx = yf.Ticker("^GSPC").history(period="1mo")["Close"]
    spx_1m_return = (spx.iloc[-1] / spx.iloc[0] - 1) * 100
    return {
        "vix": float(vix),
        "us_10y_yield_pct": float(tnx),
        "spx_1m_return_pct": float(spx_1m_return),
    }


# ---------------------------------------------------------------------------
# Nodes — each is a pure function `state -> partial_state_update`.
# ---------------------------------------------------------------------------
def fundamentals_analyst(state: CommitteeState) -> dict:
    data = fetch_fundamentals(state["ticker"])
    sys = SystemMessage(content=(
        "You are a senior equity fundamentals analyst. Given the ratios, write a SHORT "
        "report (max 150 words) covering: (1) valuation stance, (2) profitability, "
        "(3) balance-sheet strength, (4) growth. Conclude with a tilt: "
        "BULLISH / NEUTRAL / BEARISH on fundamentals."
    ))
    msg = HumanMessage(content=f"Ticker: {state['ticker']}\nRatios: {data}")
    out = sonnet.invoke([sys, msg])
    return {"fundamentals_report": out.content}


def technicals_analyst(state: CommitteeState) -> dict:
    ind = compute_indicators(state["ticker"])
    sys = SystemMessage(content=(
        "You are a technical analyst. Given price-derived indicators, write a SHORT "
        "report (max 120 words) on trend (price vs SMA50/SMA200), momentum (RSI), and "
        "volatility. Conclude with: BULLISH / NEUTRAL / BEARISH on technicals."
    ))
    msg = HumanMessage(content=f"Ticker: {state['ticker']}\nIndicators: {ind}")
    out = sonnet.invoke([sys, msg])
    return {"technicals_report": out.content}


def sentiment_analyst(state: CommitteeState) -> dict:
    headlines = fetch_news(state["ticker"])
    body = "\n- ".join(headlines) if headlines else "(no headlines available)"
    sys = SystemMessage(content=(
        "You are a news sentiment analyst. Each line below is one article, already "
        "filtered to be primarily about the ticker. Format is "
        "'[date, publisher] title — first sentence'. Use the metadata, not just the "
        "title text:\n"
        "  - Weight RECENT articles more heavily than older ones (sentiment decays).\n"
        "  - Treat tier-1 financial press (Reuters, Bloomberg, WSJ, FT) as harder "
        "evidence than opinion outlets or sector blogs.\n"
        "  - If coverage is concentrated in a single publisher, flag it as possible "
        "PR push rather than broad market sentiment.\n"
        "Produce a SHORT report (max 120 words) and score sentiment as: "
        "POSITIVE / MIXED / NEGATIVE."
    ))
    msg = HumanMessage(content=f"Ticker: {state['ticker']}\nHeadlines:\n- {body}")
    out = haiku.invoke([sys, msg])
    return {"sentiment_report": out.content}


def macro_analyst(state: CommitteeState) -> dict:
    macro = fetch_macro_snapshot()
    sys = SystemMessage(content=(
        "You are a macro strategist. Given a short macro snapshot, write a SHORT report "
        "(max 120 words) on the regime (risk-on / risk-off / neutral) and what it "
        "implies for a single-name long-only equity position."
    ))
    msg = HumanMessage(content=f"Macro snapshot: {macro}")
    out = sonnet.invoke([sys, msg])
    return {"macro_report": out.content}

def portfolio_manager(state: CommitteeState) -> dict:
    sys = SystemMessage(content=(
        "You are the Portfolio Manager. Synthesise the four analyst reports into a "
        "single thesis (max 250 words). End STRICTLY with one line of the form:\n"
        "ACTION=<BUY|HOLD|SELL>; CONFIDENCE=<0-100>; SIZE_PCT=<0.0-5.0>\n\n"
        "Where:\n"
        "  - ACTION    : the directional call.\n"
        "  - CONFIDENCE: how strongly you back the call (0 = none, 100 = very high).\n"
        "  - SIZE_PCT  : the target position as a percentage of TOTAL PORTFOLIO NAV.\n"
        "                Size purely on your conviction; do not anticipate any\n"
        "                downstream review. Use this scale:\n"
        "                  0.0      -> HOLD or SELL (no new long exposure).\n"
        "                  0.5-1.0  -> probe / starter position when conviction is mixed.\n"
        "                  1.0-2.0  -> standard BUY at moderate conviction (CONFIDENCE 50-70).\n"
        "                  2.0-3.0  -> strong BUY at high conviction (CONFIDENCE 70+).\n"
        "                  3.1-5.0  -> concentrated, high-conviction bet. Reserve for\n"
        "                              exceptionally clean setups where every analyst\n"
        "                              is meaningfully bullish AND the macro regime\n"
        "                              supports the trade.\n"
        "                A BUY at SIZE_PCT=0 is incoherent — pick HOLD instead.\n"
        "                A HOLD or SELL must have SIZE_PCT=0."
    ))
    msg = HumanMessage(content=(
        f"Ticker: {state['ticker']}\n\n"
        f"--- Fundamentals ---\n{state['fundamentals_report']}\n\n"
        f"--- Technicals ---\n{state['technicals_report']}\n\n"
        f"--- Sentiment ---\n{state['sentiment_report']}\n\n"
        f"--- Macro ---\n{state['macro_report']}"
    ))
    out = opus.invoke([sys, msg])
    return {"pm_thesis": out.content}


def _extract_action(thesis: str) -> str:
    upper = thesis.upper().replace(" ", "")
    if "ACTION=BUY" in upper:
        return "BUY"
    if "ACTION=SELL" in upper:
        return "SELL"
    return "HOLD"


def risk_officer(state: CommitteeState) -> dict:
    ind = compute_indicators(state["ticker"])
    vol = ind["vol_30d_annualized"]

    # Hard, deterministic rules — checked before any LLM call.
    if vol is not None and vol > 0.60:
        return {
            "risk_verdict": f"VETO: realized 30d vol {vol:.1%} exceeds 60% ceiling.",
            "final_decision": "HOLD (risk veto)",
        }

    sys = SystemMessage(content=(
        "You are the Chief Risk Officer. Review the PM thesis and reject if any of the "
        "following are true: (a) BUY recommended but the thesis ignores macro headwinds, "
        "(b) SIZE_PCT > 3.0 on a single name (the firm-wide single-name cap), "
        "(c) CONFIDENCE below 50. Otherwise approve. Reply STRICTLY in the form:\n"
        "VERDICT=<APPROVED|REJECTED>; REASON=<short explanation>"
    ))
    msg = HumanMessage(content=(
        f"PM thesis:\n{state['pm_thesis']}\n\nRealized 30d vol (ann.): {vol:.1%}"
    ))
    out = opus.invoke([sys, msg])
    verdict_text = out.content
    if "REJECTED" in verdict_text.upper():
        final = "HOLD (risk rejected)"
    else:
        final = _extract_action(state["pm_thesis"])
    return {"risk_verdict": verdict_text, "final_decision": final}


# ---------------------------------------------------------------------------
# Graph assembly — START fans out to four analysts, all four converge on the
# Portfolio Manager, the PM hands off to the Risk Officer, and we are done.
# ---------------------------------------------------------------------------
def build_committee():
    g = StateGraph(CommitteeState)
    g.add_node("fundamentals", fundamentals_analyst)
    g.add_node("technicals", technicals_analyst)
    g.add_node("sentiment", sentiment_analyst)
    g.add_node("macro", macro_analyst)
    g.add_node("portfolio_manager", portfolio_manager)
    g.add_node("risk_officer", risk_officer)

    for analyst in ("fundamentals", "technicals", "sentiment", "macro"):
        g.add_edge(START, analyst)
        g.add_edge(analyst, "portfolio_manager")
    g.add_edge("portfolio_manager", "risk_officer")
    g.add_edge("risk_officer", END)
    return g.compile()


def run_committee(ticker: str) -> dict:
    initial: CommitteeState = {
        "ticker": ticker,
        "fundamentals_report": "",
        "technicals_report": "",
        "sentiment_report": "",
        "macro_report": "",
        "pm_thesis": "",
        "risk_verdict": "",
        "final_decision": "",
    }
    return build_committee().invoke(initial)


if __name__ == "__main__":
    result = run_committee("TSLA")
    print("=" * 80)
    print(f"FINAL DECISION for {result['ticker']}: {result['final_decision']}")
    print("=" * 80)
    print("\n--- PM Thesis ---")
    print(result["pm_thesis"])
    print("\n--- Risk Verdict ---")
    print(result["risk_verdict"])
    print("\n\n")
    print("=" * 80)
    print("FOUR ANALYSTS DETAILS")
    print("=" * 80)
    print("\n--- Fundament Report ---")
    print(result["fundamentals_report"])
    print("\n--- Technical Report ---")
    print(result["technicals_report"])
    print("\n--- Sentiment Report ---")
    print(result["sentiment_report"])
    print("\n--- Macro Report ---")
    print(result["macro_report"])