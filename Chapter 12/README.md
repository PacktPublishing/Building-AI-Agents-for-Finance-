# Lab: Operationalising AI Agents Responsibly in Finance

Companion code for the chapter *"Operationalising AI Agents Responsibly in Finance."*
Everything runs on the **OpenAI Agents SDK** (with a small raw-API toy for teaching the
harness, and a set of offline snippet files that illustrate individual sections), from
inside this self-contained folder.

The models are `gpt-4o-mini` (the small-model tier used for the orchestrator and both sub-agents) 
with `gpt-4o` named in the prose as the escalation tier.

## Files and the chapter sections they belong to

The chapter has nine `##` sections. The table maps each file to the section(s) it serves.
The two **substrate** files and the shared helper underpin every lab, so they appear against
more than one section.

| File | Chapter section | Role |
|---|---|---|
| `fa_workflow.py` | *Engineering the agent harness* (substrate for Observability & Guardrailing) | The **fundamental-analysis workflow**: one orchestrator + two sub-agents (`FinancialDataAgent`, `NewsSentimentAgent`) wired with the agents-as-tools pattern, over two yfinance tools. The SDK *is* the harness; this is the substrate both labs run on. |
| `mini_analyst.py` | *Engineering the agent harness* + *Engineering the agentic loop* | The harness/loop **teaching toy**: the same agent with a *hand-rolled* tool loop on the raw OpenAI chat-completions API (no framework), so the tool-execution loop, state, and iteration budget are visible before the SDK hides them. |
| `snippets_harness.py` | *Engineering the agent harness* | Offline illustrations of the harness's resilience controls: retry/backoff with timeout, idempotency key, and a circuit breaker with a degradation ladder. |
| `snippets_loop.py` | *Engineering the agentic loop* | Offline illustration of a **checkpointed batch run**: a crash on name *k* resumes from *k*, not from name 1. |
| `snippets_cost.py` | *Managing cost, latency, and scaling trade-offs* | Offline illustrations of the cost/scaling levers: a run-cost estimator, a tool-result cache, a small-model→large-model cascade, and a semaphore-bounded fan-out. Imports the cost model from `tracing.py`. |
| `tracing.py` | *Observing agents in production* (shared helper) | Shared observability helpers: the OpenAI cost model (`PRICE_PER_MTOK`, `cost_usd`), ISO-duration, and the SDK-span tree renderer. Also imported by `snippets_cost.py` and `lab1_observability.py`. |
| `lab1_observability.py` | *Observing agents in production* | **Lab 1.** A custom `TracingProcessor` captures the SDK's spans; steps render the trace tree, roll runs up into per-run metrics, and detect a drift/incident signal on an injected tool failure. Writes `oai_step_#_obs_*.md` reports. |
| `snippets_deploy.py` | *Deploying and versioning agent systems* | Offline illustrations of deployment mechanics: a version manifest/fingerprint and a shadow-comparison of two versions. |
| `guardrails.py` | *Guardrailing agent inputs and outputs* | The guardrail pipeline on the SDK's native hooks: `@input_guardrail` (ticker validation), `@tool_output_guardrail` (indirect-injection screen on retrieved news), `@output_guardrail` (output policy), plus deterministic PII-redaction and disclaimer transforms. |
| `lab2_guardrails.py` | *Guardrailing agent inputs and outputs* | **Lab 2.** Assembles and red-teams the guardrail pipeline (steps 1–6). Writes `oai_step_#_guard_*.md` reports. |
| `redteam_cases.py` | *Guardrailing agent inputs and outputs* | The adversarial suite consumed by Lab 2, Step 6. |
| `snippets_governance.py` | *Governing agents: risk, compliance, and the human in the loop* | Offline illustrations of the governance mechanics: an autonomy-tier approval gate that disposes of proposed actions (execute / hold for sign-off / escalate), and an audit-record assembler that binds one decision to the trace, version, and guardrail artefacts earlier sections already produce. |
| `results/traces/` | *Observing agents in production* | JSONL span files (Lab 1), one per run. |
| `results/lab_runs/` | Observability & Guardrailing | Per-step markdown reports (`oai_step_#_obs_*.md`, `oai_step_#_guard_*.md`). |

The `snippets_*.py` files are **offline, deterministic, and free**: no API key and no network,
each finishes in seconds. The labs (`lab1_*`, `lab2_*`, `lab3_*`, `fa_workflow.py`,
`mini_analyst.py`) make live OpenAI API calls.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
cp .env.example .env                 # then put your OPENAI_API_KEY in .env
```

Only `OPENAI_API_KEY` is required. The labs run on `gpt-4o-mini`, so a full run of the tiny
watchlist costs a fraction of a cent. The coverage allow-list is in `fa_workflow.py` (`ALLOWED_TICKERS`). 
On Windows you can invoke the interpreter directly as`.venv/Scripts/python.exe` instead of activating first; the commands below use `python`.

## Running each script

### The offline snippet files (no API key, no network)

```bash
python snippets_harness.py           # retry/backoff, idempotency, circuit breaker + degradation
python snippets_loop.py              # checkpointed batch: crash on name 3, resume without re-running 1-2
python snippets_cost.py              # cost estimator, tool-result cache, model cascade, bounded fan-out
python snippets_deploy.py            # version manifest/fingerprint, shadow comparison
python snippets_governance.py        # autonomy-tier approval gate, audit-record assembler
```

### The workflow and the harness (live API)

```bash
python fa_workflow.py MSFT           # the fundamental-analysis workflow, once
python mini_analyst.py MSFT          # the hand-rolled harness/loop toy
```

### Lab 1 — observability, and Lab 2 — guardrails (live API)

```bash
python lab1_observability.py         # all of Lab 1 (steps 1-6 in sequence)
python lab2_guardrails.py            # all of Lab 2 (steps 1-6 in sequence)
```

Run a **single step** by importing its function (the unit of iteration; there is no test suite):

```bash
python -c "from lab1_observability import step_4_render_trace; step_4_render_trace()"
python -c "from lab2_guardrails import step_6_red_team; step_6_red_team()"
```

Each step prints to the console and writes an `oai_`-prefixed markdown report to
`results/lab_runs/` (`_obs_` for Lab 1, `_guard_` for Lab 2). Runs are non-deterministic, so
verdicts and timings vary slightly between runs.

## Requirements files

- `requirements.txt` — everything the chapter's labs and the offline snippets need.

yfinance is a community library, not affiliated with Yahoo; treat its data as illustrative,
not as investment advice.
