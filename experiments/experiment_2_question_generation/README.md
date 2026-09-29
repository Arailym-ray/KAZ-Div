# Experiment 2 — Diverse Grounded Kazakh Question Generation

This directory contains the code, frozen inputs, and final results for the
grounded question-generation evaluation of KAZ-Div.

## Design

- 100 Kazakh source passages
- 4 evaluated language models
- Baseline and KAZ-Div generation
- 5 repetitions per source/model/method
- 4,000 scheduled observations
- KAZ-Div candidate pool: K = 12
- semantic near-duplicate threshold: 0.95
- embedding model: BAAI/bge-m3
- stateless generation

## Directory structure

```text
experiment_2_question_generation/
├── README.md
├── STATUS.md
├── code/
│   ├── experiment2.ipynb
│   └── experiment2.py
├── data/
│   ├── experiment2_config.json
│   ├── experiment2_prompt_templates.json
│   ├── experiment2_runtime_metadata.json
│   └── experiment2_sources.csv
└── results/
    ├── experiment2_main_results.csv
    ├── experiment2_primary_summary_metrics.csv
    ├── experiment2_source_level_metrics.csv
    ├── experiment2_paired_statistics.csv
    ├── experiment2_operational_summary.csv
    ├── experiment2_failure_summary.csv
    └── experiment2_candidate_pool_summary.csv
```

## Main result files

`experiment2_main_results.csv` contains task-level outputs and operational
metadata for all 4,000 scheduled observations.

`experiment2_primary_summary_metrics.csv` contains aggregated diversity and
quality metrics by provider and method.

`experiment2_paired_statistics.csv` contains complete-source paired comparisons,
bootstrap confidence intervals, permutation p-values, and Holm-adjusted
p-values.

`experiment2_operational_summary.csv` reports generation success, token usage,
LLM calls, and latency.

`experiment2_candidate_pool_summary.csv` reports candidate-pool filtering and
semantic clustering behaviour for KAZ-Div.

## Reproducibility

The publication notebook has been stripped of execution outputs before release.
No API credentials are included. Hosted-model outputs may vary over time, so
the frozen configuration, prompt templates, model identifiers, and local-model
commit hashes are retained in the data directory.
