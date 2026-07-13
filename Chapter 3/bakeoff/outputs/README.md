# Captured bake-off outputs

Real, unedited run outputs (July 2026). Model: gpt-5-mini for all
OpenAI-backed frameworks; gemini-2.5-flash for Google ADK.

Measured token usage on the identical task:

| Framework | Input | Output | Total |
|---|---|---|---|
| Google ADK (gemini-2.5-flash) | 1,390 | 537 | 1,927 |
| PydanticAI | 1,508 | 874 | 2,382 |
| OpenAI Agents SDK | 1,606 | 1,045 | 2,651 |
| CrewAI | 1,547 | 1,223 | 2,770 |
| LangChain | 1,293 | 1,512 | 2,805 |
| AutoGen (RoundRobin team) | 20,373 | 7,575 | 27,948 |

The single-agent frameworks cluster around 2-3K tokens; AutoGen's
conversational team wrapper cost roughly 10x on the same task — the
price of message-passing transparency.

`mistral.txt` and `claude_agent_sdk.txt` are absent because those runs
require a Mistral API key and Anthropic API credits respectively; the
implementations are import-validated and follow the vendors' current
APIs.
