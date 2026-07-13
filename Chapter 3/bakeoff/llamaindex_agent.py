"""Bake-off: LlamaIndex FunctionAgent (data-centric agent design)."""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from common import (
    OPENAI_MODEL, check_api_key, compute_pe_tool_func,
    get_stock_data_tool_func, input_message, persist_framework_output,
    system_message,
)


def get_stock_data(ticker: str) -> str:
    """Get the latest stock price and EPS for a ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return f"{d.ticker}: price={d.price} eps={d.eps}"


def compute_pe_ratio(price: float, eps: float) -> float:
    """Compute the P/E ratio from price and earnings per share."""
    return compute_pe_tool_func(price, eps)


async def main() -> None:
    check_api_key()
    agent = FunctionAgent(
        name="FinancialAnalyst",
        description="Computes and compares P/E ratios",
        system_prompt=system_message,
        tools=[get_stock_data, compute_pe_ratio],
        llm=OpenAI(model=OPENAI_MODEL),
    )
    response = await agent.run(user_msg=input_message)
    final = str(response)
    print(final)
    persist_framework_output("llamaindex", final)


if __name__ == "__main__":
    asyncio.run(main())
