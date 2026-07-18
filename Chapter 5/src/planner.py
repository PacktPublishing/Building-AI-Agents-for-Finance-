"""
planner.py
==========
Research planning component for the Deep Search Agent.

Takes a research question and produces a ResearchPlan — a structured
set of sub-tasks with explicit dependencies. Uses PydanticAI's
structured output to guarantee a valid plan schema.
"""

from pydantic_ai import Agent
from common import ResearchPlan, PLANNING_MODEL

PLANNER_SYSTEM_PROMPT = """\
You are a financial research planner. Given a research question, decompose it
into concrete sub-tasks that can be executed independently by tools.

Available data sources for each sub-task:
- financial_api: Structured financial metrics (revenue, EPS, margins, P/E,
  market cap, price). Best for quantitative data.
- sec_filing: SEC filing content (risk factors, business descriptions, MD&A
  sections). Best for qualitative company-specific information.
- web_search: Recent news, competitive intelligence, analyst commentary.
  Best for current events and market context.

Rules for creating the plan:
1. Each sub-task must have a clear, measurable outcome (e.g., "Fetch NVIDIA
   revenue and margin data" not "Understand NVIDIA").
2. Specify which data_sources each sub-task should use (can be multiple).
3. Mark dependencies — which task IDs must complete before this task can start.
4. Tasks with no dependencies can execute in parallel — maximize parallelism.
5. For multi-company comparisons, create parallel tasks for each company.
6. Include a comparative analysis task that depends on all company data tasks.
7. The final task should synthesize all findings — it depends on everything.
8. Use sequential IDs starting from 1.
9. Aim for 5-10 sub-tasks. Too few = insufficient depth. Too many = overhead.
10. If the question is about a single company, still gather peer context.\
"""

# PydanticAI agent with structured output — replaces raw OpenAI API calls
planner_agent = Agent(
    model=PLANNING_MODEL,
    system_prompt=PLANNER_SYSTEM_PROMPT,
    output_type=ResearchPlan,
    retries=2,
)


async def create_research_plan(
    question: str, additional_context: str = ""
) -> ResearchPlan:
    """Generate a structured research plan from a question.

    Uses PydanticAI with structured output to produce a plan that
    can be executed by the researcher.

    Args:
        question: The research question to plan for.
        additional_context: Optional context from previous attempts
            (e.g., "Previous attempt failed to get AMD data").

    Returns:
        A ResearchPlan with sub-tasks and dependency information.
    """
    user_message = question
    if additional_context:
        user_message += f"\n\nAdditional context:\n{additional_context}"

    result = await planner_agent.run(user_message)
    plan = result.output

    # Validate the plan structure
    _validate_plan(plan)

    return plan


def _validate_plan(plan: ResearchPlan) -> None:
    """Validate that the plan has a reasonable structure.

    Checks:
    - At least 2 sub-tasks
    - All dependency IDs reference existing tasks
    - No circular dependencies (simple check)
    - At least one task has no dependencies (can start immediately)
    """
    task_ids = {t.id for t in plan.sub_tasks}

    if len(plan.sub_tasks) < 2:
        raise ValueError(
            f"Plan has only {len(plan.sub_tasks)} sub-tasks. "
            f"A research plan should have at least 2."
        )

    for task in plan.sub_tasks:
        for dep in task.dependencies:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {task.id} depends on non-existent task {dep}"
                )
            if dep == task.id:
                raise ValueError(
                    f"Task {task.id} depends on itself (circular dependency)"
                )

    startable = [t for t in plan.sub_tasks if not t.dependencies]
    if not startable:
        raise ValueError("No tasks can start — all have dependencies (deadlock)")


def format_plan(plan: ResearchPlan) -> str:
    """Format a research plan for display."""
    lines = [
        f"Research Plan: {plan.question}",
        f"Sub-tasks: {len(plan.sub_tasks)}",
        "-" * 50,
    ]

    for task in plan.sub_tasks:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
        sources = ", ".join(task.data_sources)
        lines.append(
            f"  [{task.id}] {task.description}\n"
            f"       Sources: {sources}{deps}"
        )

    return "\n".join(lines)
