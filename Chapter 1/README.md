# Chapter 1 — What Are AI Agents?

Three notebooks that build up from a plain LLM API call to a first agentic
workflow, using the same finance material throughout.

## Labs

| Lab | Notebook | What it covers | |
|---|---|---|---|
| Lab 1 | `chapter_1_lab_1_APIs_Calls_Book.ipynb` | The same earnings-call summary requested from four model families — OpenAI, Google Gemini, Anthropic Claude, and an open-weights Llama 3.1 8B Instruct loaded locally with Transformers — plus a reasoning-model call for comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%201/chapter_1_lab_1_APIs_Calls_Book.ipynb) |
| Lab 2 | `Chapter_1_lab_2_knowledge_cutoff.ipynb` | The knowledge-cutoff limitation: what a model answers when asked for news published after its training data ends | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%201/Chapter_1_lab_2_knowledge_cutoff.ipynb) |
| Lab 3 | `chapter_1_lab_3_non-agentic_vs_agentic-workflow.ipynb` | Non-agentic vs agentic on the same two questions (Nvidia's stock price, recent news): a bare LLM call, then manual tool calling, then agents built with the OpenAI Agents SDK | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PacktPublishing/Building-AI-Agents-for-Finance-/blob/main/Chapter%201/chapter_1_lab_3_non-agentic_vs_agentic-workflow.ipynb) |

All three run on Google Colab (recommended) or locally. Each notebook reads its
keys through `google.colab.userdata` and falls back to environment variables
when the Colab import fails, so the same cell works in both places. For the
local setup, copy the root [`.env.sample`](../.env.sample) to `.env` and fill in
the keys you need.

The Apple Q3 2025 earnings-call transcript used in Lab 1 is **synthetic** and
defined inline in the notebook — nothing is downloaded and no market-data API is
involved.

## API keys per lab

| Lab | Keys |
|---|---|
| Lab 1 | `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, and `HF_TOKEN` for the open-source section |
| Lab 2 | `OPENAI_API_KEY` |
| Lab 3 | `OPENAI_API_KEY`, `NEWS_API_KEY` (free tier at [newsapi.org](https://newsapi.org/register)) |

You only need the keys for the sections you actually run — each provider section
in Lab 1 is independent of the others.

## Lab 1: the open-source (Hugging Face) section

This section loads `meta-llama/Llama-3.1-8B-Instruct` on the machine running the
notebook rather than calling a hosted API, so it has three prerequisites the
API sections don't:

1. **A GPU runtime.** In Colab: *Runtime → Change runtime type → T4 GPU*.
2. **A Hugging Face token.** Create one at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and
   store it as a Colab secret named `HF_TOKEN` — `huggingface_hub` picks up a
   Colab secret of that name automatically. Locally, run `huggingface-cli login`
   or export `HF_TOKEN` in your shell.
3. **Access to the gated model.** Llama 3.1 is gated: open the
   [model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), accept
   Meta's license, and wait for the access request to be granted. Until it is,
   `from_pretrained` fails with a 401.

At `float16` the 8B weights need roughly 16 GB, which is more than a T4 has, so
`device_map="auto"` offloads part of the model to CPU RAM and generation is
slow. If it stalls or runs out of memory, either pick a runtime with a larger
GPU or swap `model_id` for a smaller instruct model such as
`meta-llama/Llama-3.2-3B-Instruct`.

## Why there is no `requirements.txt`

Each notebook installs its own packages, in the section that needs them:

| Lab | Installed in the notebook | Assumed present in Colab |
|---|---|---|
| Lab 1 | `openai`, `google-genai`, `anthropic` | `torch`, `transformers` |
| Lab 2 | `openai` | — |
| Lab 3 | `openai`, `newsapi-python`, `openai-agents` | `yfinance`, `nest_asyncio` |

Keeping the installs in the notebooks leaves one source of truth instead of two
that drift apart, and it avoids pinning `torch` and `transformers` against the
versions Colab already ships — a reliable way to break the runtime. Running
locally, install the packages in both columns for the lab you want to run.
