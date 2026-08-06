# Chapter 7 — Designing Multi-Agent Systems for Financial Applications

Five runnable labs plus the chapter's full project source.

## Labs

| Lab | Notebook | Chapter section it accompanies |
|---|---|---|
| Lab 1 | `chapter_7_lab_1_multi_agent_llamaindex.ipynb` | *Lab: Building a multi-agent financial health analyzer* — the four-agent sequential pipeline with shared state and explicit handoffs |
| Lab 2 | `chapter_7_lab_2_multi_agent_openai_sdk.ipynb` | *Implementation 2: OpenAI Agents SDK* — leader-follower routing with a manager agent and three specialist workers |
| Lab 3 | `chapter_7_lab_3_multi_agent_autogen.ipynb` | *Implementation 3: AutoGen (conversational multi-agent)* — a conversational analyst that writes and executes its own analysis code |
| Lab 4 | `chapter_7_lab_4_evaluator_optimizer.ipynb` | *Extension: Adding the evaluator-optimizer pattern* — a fifth agent that reviews the Supervisor's output and routes back for revision |
| Lab 5 | `chapter_7_lab_5_parallel_multi_company.ipynb` | *Extension: Parallel multi-company analysis* — five concurrent pipeline instances via `asyncio.gather` |

Labs 1, 2, 4, and 5 use **stub data** bundled in `common.py`, so the only key
they need is `OPENAI_API_KEY`. Lab 3's generated code fetches live market data
with `yfinance` (no key needed for the data). The `src/` tree is the
production-shaped version that calls the real Financial Modeling Prep API.

## Running a lab on Google Colab (step by step)

1. Open [colab.research.google.com](https://colab.research.google.com), choose
   **File → Open notebook → GitHub**, paste this repository's URL
   (`https://github.com/PacktPublishing/Building-AI-Agents-for-Finance-`), and
   pick the Chapter 7 notebook you want.
2. Add your OpenAI API key to Colab's Secrets: click the **key icon** in the
   left sidebar, choose **Add new secret**, name it exactly `OPENAI_API_KEY`,
   paste your key as the value, and switch on **Notebook access**.
3. Choose **Runtime → Run all**. The first cells install the packages the lab
   needs and download `common.py` from this repository automatically — there is
   nothing else to upload.
4. Watch the streamed agent banners as the pipeline hands off from agent to
   agent; the final cell prints the assessment (Lab 3 additionally displays any
   plot the generated code saved).

## Running a lab locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install jupyter llama-index llama-index-llms-openai openai-agents "pyautogen<0.4" yfinance matplotlib pydantic python-dotenv
export OPENAI_API_KEY="your_key"
jupyter notebook
```

Python 3.11+ recommended. Each notebook also installs its own dependencies in
its first cell, so installing just `jupyter` and running the notebook works too.

## Project source (`src/`)

Install its pinned dependencies first and run the modules from this
folder:

```bash
cd "Chapter 7"
pip install -r src/requirements.txt
```

`src/` is the chapter's project code as printed in the implementation
sections — unlike the notebooks, it calls the real Financial Modeling Prep
API, so it needs both `OPENAI_API_KEY` and `FINANCIAL_MODELING_PREP_API_KEY`
in the environment (or in a `.env` file):

- `src/llamaindex/` — the four-agent sequential pipeline
  (`agents.py`, `tools.py`, `run.py`); run with
  `python -m src.llamaindex.run AAPL`
- `src/openai_sdk/` — the leader-follower implementation with
  agents-as-tools synthesis; run with `python -m src.openai_sdk.run`
- `src/autogen/` — the conversational analyst (classic AutoGen 0.2 API);
  run with `python -m src.autogen.run`
- `src/extensions/evaluator.py` — the EvaluatorAgent and its deterministic
  quality-check tool (the chapter's evaluator-optimizer extension)
- `src/extensions/parallel_analysis.py` — the parallel portfolio runner;
  run with `python -m src.extensions.parallel_analysis`
- `src/common.py` — shared thresholds, prompt helpers, and configuration
- `src/requirements.txt` — pinned dependencies for all three implementations

## Note on iteration limits and termination

The four- and five-agent pipelines need more workflow iterations than
LlamaIndex's default (`max_iterations=20`): each agent spends roughly three
iterations on its tool call, handoff, and response, and any retry burns more.
The notebooks therefore run the workflow with `max_iterations=60`, keep every
agent to a single tool call, and make the last agent write the final answer
instead of handing off again. Lab 3's termination discipline is the
conversational analogue: the assistant may only say `TERMINATE` after it has
seen execution results, and `max_turns=5` provides a hard stop.

## Note on the AutoGen version

Lab 3 and `src/autogen/` use the classic AutoGen 0.2-style API
(`pip install "pyautogen<0.4"`), the most widely deployed and documented
lineage. AutoGen 0.4 was a ground-up rewrite, and Microsoft has since folded
the project's direction into the Microsoft Agent Framework; the community
fork AG2 continues the 0.2 lineage. Chapter 3's AutoGen coverage picks up the
framework-succession story.
