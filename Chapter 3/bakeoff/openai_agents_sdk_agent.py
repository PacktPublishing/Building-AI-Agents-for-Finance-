"""Bake-off: OpenAI Agents SDK (sequential tool calling)."""

from agents import Agent, Runner, function_tool

from common import (
    OPENAI_MODEL, check_api_key, compute_pe_tool_func,
    get_stock_data_tool_func, input_message, persist_framework_output,
    system_message,
)


@function_tool
def get_stock_data(ticker: str) -> str:
    """Get the latest stock price and EPS for a ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return f"{d.ticker}: price={d.price} eps={d.eps}"


@function_tool
def compute_pe_ratio(price: float, eps: float) -> float:
    """Compute the P/E ratio from price and earnings per share."""
    return compute_pe_tool_func(price, eps)


def main() -> None:
    check_api_key()
    agent = Agent(
        name="Financial Analysis Agent",
        instructions=system_message,
        tools=[get_stock_data, compute_pe_ratio],
        model=OPENAI_MODEL,
    )
    print("Running agent...")
    result = Runner.run_sync(agent, input_message)
    final_response = (
        result.final_output if hasattr(result, "final_output")
        else str(result)
    )
    usage = ""
    try:
        u = result.context_wrapper.usage
        usage = (f"input={u.input_tokens} output={u.output_tokens} "
                 f"total={u.total_tokens}")
    except AttributeError:
        pass
    print(final_response)
    if usage:
        print(f"\n[token usage] {usage}")
    persist_framework_output("openai_agents_sdk", final_response, usage)


if __name__ == "__main__":
    main()
