# KYC Evaluation Dataset

Sample dataset for the Chapter 11 evaluation harness. The full dataset
(~2,050 cases — see the chapter's The KYC evaluation dataset
structure section) is built from production
archives, synthetic generation, and red-team campaigns. The starter
dataset shipped here demonstrates the schema and exercises every code
path in the harness.

## Files

| File | Cases | Purpose |
|------|-------|---------|
| `core_set.jsonl` | 5 | Stratified by risk tier; primary task-success metric |
| `sanctions_set.jsonl` | 5 | Confirmed matches + name-similar non-matches |
| `adversarial_set.jsonl` | 5 | Prompt injection, edge cases, edge-case identity |
| `disparate_impact_set.jsonl` | 6 | Demographically-balanced for fair-treatment analysis |
| `calibration_set.jsonl` | 5 | Human-labeled grounding examples for LLM-judge calibration |

## Schema

Each line is a JSON object conforming to `src.models.KYCCase`. See that
file for the full Pydantic schema. Fields:

- `case_id` — unique identifier; segment-prefixed (e.g. `core-001`)
- `segment` — one of `core`, `sanctions`, `adversarial`, `disparate_impact`, `regression`
- `customer` — `CustomerProfile` (name, DOB, jurisdiction, optional free-text)
- `documents` — list of `DocumentReference`
- `expected` — `KYCExpectedOutcome` with gold risk tier, sanctions match flag, required tools, rationale
- `metadata` — provenance: source, created_at, labelers, kappa

## Production dataset construction

For the production harness, you will need:

- **Core set: ~1,000 cases** stratified by risk tier (250 per tier),
  drawn from anonymized historical archives with confirmed disposition
  outcomes
- **Sanctions set: ~200 cases** (100 confirmed matches, 100
  name-similar non-matches), generated from real OFAC list patterns
- **Adversarial set: ~150 cases** (50 prompt injections, 50 document
  spoofs, 50 identity edge cases) curated by a red team
- **Disparate-impact set: ~500 cases** stratified across demographic
  groups for statistical significance
- **Live regression set: ~200 cases** drawn from anonymized production
  traffic, refreshed quarterly

## Labeling protocol

- Two compliance officers label each case independently
- A senior reviewer adjudicates disagreements
- Inter-rater reliability (Cohen's kappa) measured on every batch
- Minimum kappa: 0.7 for production-grade datasets

## Versioning

Datasets are versioned in git. Updates create new versions rather than
mutating cases in place. Each case's `metadata.label_version` field
records which version of the labeling protocol produced its labels.
