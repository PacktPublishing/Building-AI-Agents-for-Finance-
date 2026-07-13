# Building AI Agents for Finance

Code repository for *Building AI Agents for Finance* (Packt, 2026).

Each chapter folder contains one or more hands-on Jupyter notebooks
named `chapter_<N>_lab_<M>_<topic>.ipynb`. Several chapters also ship
a `common.py` with shared models, helpers, and sample data that the
notebooks import; the notebooks auto-download `common.py` on first
run in Colab.

## Running the labs

### In Google Colab (recommended)

Each notebook is designed to run end-to-end in Colab. Open the
notebook, and when prompted for an API key, add it via the **key**
icon in Colab's left sidebar using the exact name listed in the
notebook (e.g. `OPENAI_API_KEY`). The notebooks read keys with
`google.colab.userdata.get(...)`.

### Locally

1. Copy [`.env.sample`](./.env.sample) to `.env` and fill in the keys
   you need (each section lists which labs use which key).
2. Install the per-lab packages with the `%pip install` cell at the
   top of each notebook.
3. Open the notebook in Jupyter / VS Code / your editor of choice.

The notebooks wrap the Colab import in `try / except ImportError` and
fall back to reading from the environment when running locally.

## Repository layout

```text
Chapter 1/        First-principles labs (Hanane)
Chapter 2/        Memory and tool patterns (Hanane)
Chapter 3/        Framework comparison — one lab notebook per framework,
                  plus bakeoff/ (the chapter's printed implementations with
                  captured outputs and pinned requirements), templates/
                  (decision-matrix spreadsheet and scoring script), and
                  diagrams/ (figure sources)
Chapter 5/        Deep search agent (planner + full pipeline)
Chapter 7/        Multi-agent systems and architectural styles
Chapter 9/        Insurance workflows (claims, fraud, underwriting)
.env.sample       Template for local-execution API keys
```

## Provider keys at a glance

| Key | Required by |
| --- | --- |
| `OPENAI_API_KEY` | Ch 1, 2, 3, 5, 7, 9 (most labs) |
| `ANTHROPIC_API_KEY` | Ch 1 Lab 1, Ch 3 Lab 7 |
| `GOOGLE_API_KEY` | Ch 1 Lab 1, Ch 3 Lab 2 |
| `MISTRAL_API_KEY` | Ch 3 Lab 8 |
| `NEWS_API_KEY` | Ch 1 Lab 3 |
| `FINANCIAL_MODELING_PREP_API_KEY` | Ch 2 Labs 1 and 6 |
| `FINANCIAL_DATASETS_API_KEY`, `TAVILY_API_KEY`, `SEC_EDGAR_EMAIL` | Only if you swap Ch 5's stub tools for the real ones |

See [`.env.sample`](./.env.sample) for sources and per-key notes.
