# Experiment 4 — Stage-Wise Ablation Analysis

This directory contains post-hoc stage-wise ablation results for Experiment 1 (Kazakh paraphrase generation). No new LLM generations were required for the ablation.

## Conditions

- **B0 — Standard Baseline:** one direct response.
- **B1 — Multi-Candidate Random:** uniform random selection from the raw multi-candidate pool.
- **B2 — Quality-Filtered Random:** quality filtering followed by uniform random selection from valid candidates; no exact or semantic deduplication.
- **B3 — Full KAZ-Div:** the complete KAZ-Div pipeline with quality filtering, semantic deduplication/clustering, representative extraction, and cluster-balanced selection.

## Result groups

### `api_models/`
Results for:
- GPT-5.6 Sol (`openai`)
- Claude Sonnet 4.6 (`anthropic`)

### `local_models/`
Final frozen-quality results for:
- Meta-Llama-3.1-8B-Instruct (`llama`)
- LLama-3.1-KazLLM-1.0-8B (`kazllm`)

For the final local-model run, B2 reuses the candidate-level quality decisions saved in Experiment 1 `quality_metadata`, avoiding threshold-edge differences from recomputing BGE-M3 fidelity.

## Statistical analysis

Primary comparisons are the planned paired contrasts:

1. **B1 − B0:** contribution of multi-candidate generation.
2. **B2 − B1:** contribution of quality filtering.
3. **B3 − B2:** additional contribution of semantic curation and cluster-balanced selection.

The analysis uses 10,000 paired bootstrap resamples, 20,000 paired sign-flip permutations, and Holm correction across the 21 planned tests within each model.

## Files

- `experiment4_ablation_observations.csv` — observation-level B0–B3 outputs.
- `experiment4_source_level_metrics.csv` — source-level diversity and fidelity metrics.
- `experiment4_complete_case_metrics.csv` — complete-case source-level analysis data.
- `experiment4_primary_summary_metrics.csv` — primary B0–B3 summary table.
- `experiment4_paired_statistics.csv` — planned paired contrasts, confidence intervals, permutation p-values, and Holm-adjusted p-values.
- `experiment4_operational_summary.csv` — operational success and fidelity summaries.
- `experiment4_candidate_pool_diagnostics_request_level.csv` — request-level candidate-pool diagnostics.
- `experiment4_candidate_pool_diagnostics_summary.csv` — aggregated candidate-pool diagnostics.
- `experiment4_run_manifest.json` — frozen experiment configuration and analysis metadata.

The central ablation finding is that multi-candidate generation accounts for most of the observed increase in output diversity, while subsequent KAZ-Div stages primarily provide quality control and semantic redundancy management.

## Supplementary Materials

- `supplementary_table_S14_stage_wise_ablation.csv` / `.md` — full descriptive B0–B3 results for all four models.
- `supplementary_table_S15_planned_paired_contrasts.csv` / `.md` — all planned paired contrasts (B1−B0, B2−B1, B3−B2) with mean differences, 95% paired-bootstrap confidence intervals, raw permutation-test p-values, and Holm-adjusted p-values.
- `section_4_4_supplementary_reference.md` — recommended sentence linking the main Results section to Tables S14–S15.

These table numbers intentionally follow the existing Supplementary Tables S1–S13, avoiding renumbering of earlier materials.
