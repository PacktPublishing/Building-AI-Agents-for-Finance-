# Chapter 2 — Exploring Design Patterns for AI Agents

Six notebooks, one per design pattern: tool use, working memory, episodic
memory, ReAct, evaluator-optimizer, and a sequential (prompt-chaining)
workflow — all applied to the same kind of finance material.

## Labs

| Lab | Notebook | What it covers | |
|---|---|---|---|
| Lab 1 | `chapter_2_lab_1_retrieving_fundamental_ratios_Apple.ipynb` | **Tool use.** An OpenAI Agents SDK agent (`gpt-4o-mini`) calls a `get_company_fundamentals` tool built on Finance Toolkit / Financial Modeling Prep to return Apple's profitability, valuation, and liquidity ratios — and shows how the model's knowledge cutoff decides which year it treats as "the most recent fiscal year" | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_1_retrieving_fundamental_ratios_Apple.ipynb) |
| Lab 2 | `chapter_2_lab_2_portfolio_rebalancing_working_memory.ipynb` | **Working memory.** A portfolio-rebalancing agent (`gpt-4o-mini`) carried across six turns with an Agents SDK `SQLiteSession`, printing the stored conversation after each turn, then a first pass at condensing it into an episode | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_2_portfolio_rebalancing_working_memory.ipynb) |
| Lab 3 | `chapter_2_lab_3_portfolio_rebalancing_episodic_memory.ipynb` | **Episodic memory.** The same conversation turned into a structured JSON episode (summary, allocation trajectory, what worked / what didn't, key points) two ways — a LangChain chain and the OpenAI Responses API — embedded with `text-embedding-3-small`, stored in ChromaDB, retrieved by similarity, and injected into a fresh session; ends with an interactive loop that writes the episode when you type `exit` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_3_portfolio_rebalancing_episodic_memory.ipynb) |
| Lab 4 | `chapter_2_lab_4_ReAct_agent_LlamaIndex_fetching_stock_price_information.ipynb` | **ReAct.** LlamaIndex's workflow `ReActAgent` (`gpt-4o`) with a `get_latest_price` tool over Yahoo Finance, streamed so the Thought / Action / Observation trace is visible, then re-run with a `Context` so a follow-up question is answered from memory instead of a second tool call | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_4_ReAct_agent_LlamaIndex_fetching_stock_price_information.ipynb) |
| Lab 5 | `chapter_2_lab_5_Financial_News_agent_evaluator_optimizer_pattern.ipynb` | **Evaluator-optimizer (LLM as a judge).** `web_news_searcher` (hosted `WebSearchTool`) collects recent Reuters headlines for a region; `news_evaluator` returns a `feedback` + `score` dataclass; the loop re-runs the searcher with the feedback until the score is `successful` or four iterations pass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_5_Financial_News_agent_evaluator_optimizer_pattern.ipynb) |
| Lab 6 | `chapter_2_lab_6_investment_recommendation_sequential_pattern.ipynb` | **Sequential workflow (prompt chaining).** Three `gpt-4o-mini` agents in a fixed chain — retrieval (FMP ratios) → analysis (profitability, valuation, risks) → decision (BUY / HOLD / SELL plus a watchlist) — each handing structured JSON to the next | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%202/chapter_2_lab_6_investment_recommendation_sequential_pattern.ipynb) |

Labs 2 and 3 share the same agent and the same six-turn conversation. Lab 3 is
the fuller treatment of long-term memory; if you only run one of the two, run
Lab 3.

## Running the labs

These notebooks are written for Google Colab and read their keys from the
Colab secret store:

```python
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
```

Add each key under the **key** icon in Colab's left sidebar, using exactly the
names in the table below.

To run locally, copy the root [`.env.sample`](../.env.sample) to `.env`, fill
in the keys you need, and replace that cell with:

```python
import os
from dotenv import load_dotenv          # pip install python-dotenv

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Labs 1 and 6 also need:
# FINANCIAL_MODELING_PREP_API_KEY = os.environ["FINANCIAL_MODELING_PREP_API_KEY"]
```

## API keys per lab

| Lab | Keys |
|---|---|
| Lab 1 | `OPENAI_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY` |
| Lab 2 | `OPENAI_API_KEY` (chat completions and embeddings) |
| Lab 3 | `OPENAI_API_KEY` (chat completions and embeddings) |
| Lab 4 | `OPENAI_API_KEY` |
| Lab 5 | `OPENAI_API_KEY` |
| Lab 6 | `OPENAI_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY` |

Yahoo Finance (Lab 4) needs no key. Lab 5 needs no separate search key either —
`WebSearchTool` is OpenAI's own hosted tool, billed per search on top of the
tokens, and it runs through the Responses API.

## Labs 1 and 6: the Financial Modeling Prep key

Both labs pull their ratios through
[Finance Toolkit](https://www.jeroenbouma.com/projects/financetoolkit), which
wraps the [Financial Modeling Prep](https://site.financialmodelingprep.com/)
API. Create a free account there; the key appears in your dashboard.

`Toolkit([ticker], api_key=..., start_date="2022-01-01")` followed by
`collect_all_ratios()` downloads the income statement, balance sheet, and cash
flow statement for every year since `start_date` and derives 37 ratios from
them, returned as a DataFrame indexed by ratio name. The labs then keep only
the profitability, valuation, and liquidity rows.

Free FMP plans are rate-limited and do not cover every statement endpoint, so
`collect_all_ratios()` can come back partly empty (a ratio row full of `NaN`
usually means the underlying statement line was not returned, not that the
company reported nothing). If that happens, check what your plan includes
before debugging the agent — the tool is fine, the data is missing.

## Labs 2 and 3: what actually persists

- `SQLiteSession("conversation_ptf")` keeps the conversation **in memory
  only** — `db_path` defaults to `":memory:"`, so the session is gone when the
  kernel restarts. Pass a file to keep it across runs:
  `SQLiteSession("conversation_ptf", "ptf_rebal.db")`.
- ChromaDB is the opposite: `PersistentClient(path="./chroma_ptf_rebal")`
  writes a folder next to the notebook. In Colab that folder lives on the
  runtime's disk and disappears when the runtime is recycled — mount Google
  Drive or download the folder if you want your episodes to survive.
- The final "All in" loop in Lab 3 reads from `input()`. It stores nothing
  until you type `exit`: that is what triggers the summarization of the
  session into an episode and the write to ChromaDB.

## Lab 4: keep the `-U` on the LlamaIndex install

The lab uses the workflow-based agent — `from
llama_index.core.agent.workflow import ReActAgent`, driven with `await
agent.run(...)` — which is why the install cell is `!pip install -U
llama-index`. On older `llama-index-core` releases (0.12.x, which is what you
get if something else in the environment pins it) that class requires `name`
and `description`, and the constructor call as written raises a Pydantic
`ValidationError`. Either upgrade, or add `name="price_agent",
description="Fetches the latest stock price"`.

## Lab 5: the loop is interactive

`main()` starts with `msg = input("User's request:")`, so run it in an
interactive session rather than as a script. The requests used in the chapter
are quoted in the comments under each run cell, for example:

> Give me the latest 5 news items in the Eurozone, and specify the source for
> each one.

The loop exits when the evaluator returns `score == "successful"`, or after
`max_iteration = 4` refinement rounds. Because the dates in the instructions
are computed from `datetime.now()`, the news window moves with the day you run
it — the output will not match the one printed in the chapter.

## Why there is no `requirements.txt`

Each notebook installs its own packages, in the section that needs them:

| Lab | Installed in the notebook | Assumed present in Colab |
|---|---|---|
| Lab 1 | `openai-agents`, `financetoolkit` | `pandas` |
| Lab 2 | `openai-agents`, `langchain_openai`, `chromadb` | `nest_asyncio` |
| Lab 3 | `openai-agents`, `langchain_openai`, `chromadb` | `nest_asyncio` |
| Lab 4 | `llama-index` (with `-U`) | `yfinance` |
| Lab 5 | `openai`, `openai-agents` | `nest_asyncio` |
| Lab 6 | `openai`, `openai-agents`, `financetoolkit` | `nest_asyncio`, `pandas` |

Keeping the installs in the notebooks leaves one source of truth instead of two
that drift apart, and it avoids pinning packages against the versions Colab
already ships. Running locally, install the packages in both columns for the
lab you want to run.

## Known rough edges

Three cells need a small manual fix if you run a notebook top to bottom:

- **Lab 3** calls `json.loads` / `json.dumps` in the "Method 2: Responses API"
  cells, but the only `import json` in the notebook is inside a commented-out
  block. Add `import json` before running them.
- **Lab 3**, same section, assigns `episodic_memory = json.dumps(...)` — a
  *string* — and the "Add Memory in DB" cell then subscripts it like a dict.
  Keep the parsed object (`json.loads(...)`) if you ran Method 2, or re-run the
  LangChain cell, which already returns a dict.
- **Lab 5** opens with a cell that displays `web_news_searcher_BOT.png` from an
  undefined `path` variable. The image is the architecture diagram from the
  chapter and is not in the repo — skip the cell.
