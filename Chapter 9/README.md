# Chapter 9 — Building Multi-Agent Insurance Workflows

Three runnable labs plus the chapter's full project source.

## Labs

| Lab | Notebook | Chapter section it accompanies |
|---|---|---|
| Lab 1 | `chapter_9_lab_1_claims_pipeline.ipynb` | *Lab: Building a claims processing pipeline* — the seven-agent sequential claims pipeline with a compliance guardrail |
| Lab 2 | `chapter_9_lab_2_fraud_investigation.ipynb` | *Advanced pattern: Multi-agent fraud investigation* — the adversarial fraud debate (prosecutor, defender, decision agent) |
| Lab 3 | `chapter_9_lab_3_underwriting.ipynb` | The underwriting extension — a sequential workflow whose enrichment stage fans out in parallel |

All three labs run on Google Colab or locally (Python 3.11+). Each notebook
installs its own dependencies (`llama-index`, `llama-index-llms-openai`,
`pydantic`, `python-dotenv`) and needs an `OPENAI_API_KEY`.

`common.py` holds the shared Pydantic models, sample claim data, and helpers
(`log_audit`, `days_between`) used by all three labs.

## Project source (`src/`)

`src/` is the chapter's project code as printed in the implementation
sections:

- `src/claims_pipeline/` — agents, tools, and runner for the claims pipeline
  (`agents.py`, `tools.py`, `run.py`, plus a captured `sample_output.txt`)
- `src/fraud_investigation/` — the adversarial fraud-debate agents
- `src/underwriting/` — the parallel-enrichment underwriting runner
- `src/models.py`, `src/common.py` — shared data contracts and configuration

## Note on Lab 1 iteration limits

The seven-agent pipeline needs more workflow iterations than LlamaIndex's
default (`max_iterations=20`): seven agents each spend roughly three
iterations (tool call, handoff, response). The notebook therefore runs the
workflow with `max_iterations=60`, keeps every agent to a single tool call,
and makes the compliance agent terminate with a final summary instead of
handing control back up the pipeline.
