"""Bake-off: AutoGen 0.4+ (conversational coordination)."""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from common import (
    OPENAI_MODEL, check_api_key, compute_pe_tool_func,
    get_stock_data_tool_func, input_message, persist_framework_output,
    system_message,
)


def get_stock_data(ticker: str) -> str:
    """Get the latest stock price and EPS for a ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return f"{d.ticker}: price={d.price} eps={d.eps}"


async def main() -> None:
    check_api_key()
    model_client = OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    get_stock_data_tool = FunctionTool(
        get_stock_data,
        description="Get stock price and EPS for a ticker (e.g., AAPL, JPM)",
    )
    compute_pe_tool = FunctionTool(
        compute_pe_tool_func,
        description="Compute P/E ratio given price and EPS",
    )
    analyst_agent = AssistantAgent(
        name="FinancialAnalyst",
        model_client=model_client,
        tools=[get_stock_data_tool, compute_pe_tool],
        system_message=system_message,
        reflect_on_tool_use=True,
    )
    team = RoundRobinGroupChat(
        participants=[analyst_agent],
        termination_condition=MaxMessageTermination(10),
    )
    print(f"Running AutoGen analysis: {input_message}\n")
    result = await team.run(task=input_message)

    in_tok = out_tok = 0
    lines = []
    for msg in result.messages:
        usage = getattr(msg, "models_usage", None)
        if usage:
            in_tok += usage.prompt_tokens
            out_tok += usage.completion_tokens
        content = getattr(msg, "content", "")
        lines.append(f"[{msg.source}] {content}")
    final = str(getattr(result.messages[-1], "content", ""))
    usage_s = f"input={in_tok} output={out_tok} total={in_tok + out_tok}"
    print("\n".join(lines[1:]))
    print(f"\n[token usage] {usage_s}")
    persist_framework_output("autogen", "\n\n".join(lines[1:]), usage_s)
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
