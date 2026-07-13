"""Bake-off: CrewAI (role-based collaboration)."""

from crewai import Agent, Crew, Task
from crewai.tools import tool

from common import (
    OPENAI_MODEL, check_api_key, compute_pe_tool_func,
    get_stock_data_tool_func, input_message, persist_framework_output,
    system_message,
)


@tool("Get stock data")
def get_stock_data(ticker: str) -> str:
    """Get the latest stock price and EPS for a ticker symbol."""
    d = get_stock_data_tool_func(ticker)
    return f"{d.ticker}: price={d.price} eps={d.eps}"


@tool("Compute P/E ratio")
def compute_pe_ratio(price: float, eps: float) -> float:
    """Compute the P/E ratio from price and earnings per share."""
    return compute_pe_tool_func(price, eps)


def main() -> None:
    check_api_key()
    analyst = Agent(
        role="Financial Analyst",
        goal="Compare stocks on valuation and write a short memo",
        backstory=system_message,
        tools=[get_stock_data, compute_pe_ratio],
        llm=OPENAI_MODEL,
        verbose=False,
    )
    task = Task(
        description=input_message,
        expected_output="A short comparative investment memo with both P/E ratios",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task])
    result = crew.kickoff()
    final = str(result)
    usage = ""
    metrics = getattr(crew, "usage_metrics", None)
    if metrics:
        usage = (f"input={metrics.prompt_tokens} "
                 f"output={metrics.completion_tokens} "
                 f"total={metrics.total_tokens}")
    print(final)
    if usage:
        print(f"\n[token usage] {usage}")
    persist_framework_output("crewai", final, usage)


if __name__ == "__main__":
    main()
