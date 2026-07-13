"""Bake-off: Google ADK (enterprise cloud-native platform).

Runs against the Gemini API with GOOGLE_API_KEY (set
GOOGLE_GENAI_USE_VERTEXAI=FALSE for API-key auth).
"""

import asyncio
import os

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from common import (
    compute_pe_tool_func, get_stock_data_tool_func, input_message,
    persist_framework_output, system_message,
)

MODEL = os.getenv("BAKEOFF_GEMINI_MODEL", "gemini-2.5-flash")


def get_stock_data(ticker: str) -> str:
    """Get the latest stock price and EPS for a ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return f"{d.ticker}: price={d.price} eps={d.eps}"


def compute_pe_ratio(price: float, eps: float) -> float:
    """Compute the P/E ratio from price and earnings per share."""
    return compute_pe_tool_func(price, eps)


async def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set; skipping live run.")
    agent = Agent(
        name="financial_analyst",
        model=MODEL,
        instruction=system_message,
        tools=[get_stock_data, compute_pe_ratio],
    )
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="bakeoff")
    content = types.Content(
        role="user", parts=[types.Part(text=input_message)])
    final = ""
    in_tok = out_tok = 0
    async for event in runner.run_async(
            user_id="bakeoff", session_id=session.id, new_message=content):
        meta = getattr(event, "usage_metadata", None)
        if meta:
            in_tok += meta.prompt_token_count or 0
            out_tok += meta.candidates_token_count or 0
        if event.is_final_response() and event.content:
            final = "".join(p.text or "" for p in event.content.parts)
    usage = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print(final)
    print(f"\n[token usage] {usage}")
    persist_framework_output("google_adk", final, usage)


if __name__ == "__main__":
    asyncio.run(main())
