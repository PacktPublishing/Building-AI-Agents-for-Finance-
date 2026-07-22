"""mini_analyst.py: The agent harness, by hand, on the raw OpenAI chat-completions API.

The labs in this chapter run on the OpenAI Agents SDK, whose Runner is a production harness
that hides the tool-execution loop from you. This file exists for the opposite reason: to
make that loop 'visible'. It is a single-file fundamental-analysis agent with a hand-rolled
tool loop written directly against `client.chat.completions.create`, no framework, so
that the harness's responsibilities (the loop, the conversation state, the tool dispatch,
the termination conditions, the iteration budget, the tool permission layer) are all in
plain sight. Read it once to understand what a harness is; then let the Agents SDK provide
one for you in the labs.

The agent answers one question : Is this stock fundamentally cheap or expensive?. It uses
two read-only tools over yfinance, and ends its answer with a machine-readable trailer:

    VERDICT: UNDERVALUED | FAIRLY_VALUED | OVERVALUED | INSUFFICIENT_EVIDENCE
    CONFIDENCE: LOW | MEDIUM | HIGH
"""

from __future__ import annotations

import json

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"                 # the SLM tier
MAX_ITERATIONS = 6                    # loop budget: a risk limit, not a nicety
MAX_TOKENS = 700

# Tool permission layer: the agent may only see tickers on this coverage allow-list. A tool
# is a capability grant; an allow-list is how we bound the blast radius of that grant.
ALLOWED_TICKERS = {"MSFT", "AAPL", "GOOGL", "JPM", "KO", "XOM", "NVDA", "AMZN"}

SYSTEM_PROMPT = """You are a buy-side equity analyst assistant. Given a ticker, decide \
whether the stock looks fundamentally undervalued, fairly valued, or overvalued, using ONLY \
the tools provided. Gather valuation ratios and recent news before concluding.

If the tools do not return enough evidence to judge (missing ratios, no news), do not guess \
-- return INSUFFICIENT_EVIDENCE. In finance an honest "not enough evidence" is a correct and \
reportable answer.

End every response with exactly two trailer lines and nothing after them:
VERDICT: <UNDERVALUED|FAIRLY_VALUED|OVERVALUED|INSUFFICIENT_EVIDENCE>
CONFIDENCE: <LOW|MEDIUM|HIGH>"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_key_ratios",
        "description": "Get valuation and profitability ratios for a ticker: trailing/forward "
                       "P/E, price-to-book, profit margin, return on equity. Use this first.",
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string", "description": "e.g. MSFT"}},
                       "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "get_recent_news",
        "description": "Get recent news headlines for a ticker, to sense-check the ratios.",
        "parameters": {"type": "object",
                       "properties": {"ticker": {"type": "string", "description": "e.g. MSFT"}},
                       "required": ["ticker"]}}},
]


# --- Tool implementations (read-only; safe to retry) -------------------------------------
def _check_ticker(ticker: str) -> str:
    """Enforce the ticker allow-list before any tool touches an external system."""
    t = (ticker or "").strip().upper()
    if t not in ALLOWED_TICKERS:
        raise PermissionError(f"ticker {t!r} is not on the coverage allow-list")
    return t


def get_key_ratios(ticker: str) -> dict:
    """Return a small, fixed set of valuation ratios. Read-only, so retry-safe."""
    t = _check_ticker(ticker)
    info = yf.Ticker(t).info
    return {"ticker": t, "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"), "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"), "return_on_equity": info.get("returnOnEquity")}


def get_recent_news(ticker: str, limit: int = 5) -> dict:
    """Return recent headlines. Read-only, so retry-safe."""
    t = _check_ticker(ticker)
    items = yf.Ticker(t).news or []
    headlines = []
    for item in items[:limit]:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if title:
            headlines.append(title)
    return {"ticker": t, "headlines": headlines}


TOOL_IMPLS = {"get_key_ratios": get_key_ratios, "get_recent_news": get_recent_news}


def run_tool(name: str, arguments: dict) -> str:
    """Execute one tool call. A failed tool still returns a result string, so the model can
    recover -- dropping it would leave a dangling tool call and break the next turn."""
    try:
        return json.dumps(TOOL_IMPLS[name](**arguments))
    except Exception as exc:  # noqa: BLE001 -- surface any failure back to the model
        return json.dumps({"error": str(exc)})


# --- The hand-rolled agentic loop --------------------------------------------------------
def analyse(ticker: str, client: OpenAI | None = None) -> dict:
    """Run the agent on one ticker. This IS the harness: it owns the conversation state, the
    tool-execution loop, the termination conditions, and the iteration budget."""
    client = client or OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyse {ticker}."}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS, tools=TOOLS, messages=messages,
        )
        message = response.choices[0].message
        messages.append(message)

        # Termination condition: the model stopped asking for tools.
        if not message.tool_calls:
            text = message.content or ""
            return {"ticker": ticker, "iterations": iteration, "answer": text,
                    "verdict": _parse_trailer(text, "VERDICT"),
                    "confidence": _parse_trailer(text, "CONFIDENCE"), "stopped": "end_turn"}

        # Otherwise execute every requested tool and feed the results back, one per call.
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = run_tool(call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    # Budget-exhausted termination: the loop hit MAX_ITERATIONS without concluding.
    return {"ticker": ticker, "iterations": MAX_ITERATIONS, "answer": "",
            "verdict": "INSUFFICIENT_EVIDENCE", "confidence": "LOW", "stopped": "max_iterations"}


def _parse_trailer(text: str, label: str) -> str:
    """Pull a machine-readable trailer value (VERDICT / CONFIDENCE) out of the answer."""
    for line in reversed((text or "").splitlines()):
        if line.strip().upper().startswith(f"{label}:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    result = analyse(ticker)
    print(f"[{result['ticker']}] verdict={result['verdict']} "
          f"confidence={result['confidence']} iterations={result['iterations']} "
          f"stopped={result['stopped']}")
    print("-" * 60)
    print(result["answer"])
