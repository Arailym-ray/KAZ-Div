# KAZ-Div

**KAZ-Div: A Stateless Model-Agnostic Framework for Quality-Constrained Diversity in Kazakh Language Generation**

This repository contains reproducibility code, frozen experimental inputs,
result files, and supplementary materials for three complementary evaluations
of KAZ-Div.

## Experiments

### 1. Kazakh paraphrase generation
Evaluates lexical, structural, and semantic diversity while preserving source
meaning and linguistic quality.

Status: **complete**.

### 2. Diverse grounded Kazakh question generation
Evaluates whether repeated generation covers different answerable aspects of a
source passage while maintaining grounding and answerability.

Status: **complete**. The final run contains 4,000 scheduled observations.

### 3. Synthetic Kazakh intent data and downstream classification
Evaluates whether diversity-controlled synthetic training data improves
downstream intent classification on an independent human-written test set.

Status: **complete**.

## Evaluated language models

- OpenAI GPT-5.6 Sol
- Anthropic Claude Sonnet 4.6
- `issai/LLama-3.1-KazLLM-1.0-8B`
- `NousResearch/Meta-Llama-3.1-8B-Instruct`

The purpose of the study is not to rank the language models. The models provide
heterogeneous proprietary and open-weight settings in which the relative effect
of Baseline versus KAZ-Div is evaluated.

## Repository structure

```text
KAZ-Div/
├── README.md
├── CITATION.cff
├── requirements.txt
├── MANIFEST.csv
└── experiments/
    ├── experiment_1_paraphrase/
    ├── experiment_2_question_generation/
    │   ├── code/
    │   ├── data/
    │   └── results/
    └── experiment_3_intent_classification/
```

## KAZ-Div procedure

For each request, KAZ-Div uses a temporary candidate pool and applies
multi-candidate generation, task-specific quality filtering, exact
deduplication, semantic near-duplicate clustering, representative selection,
and final diversity-aware selection.

The framework is stateless across independent requests: previous generated
outputs, previous candidate pools, and conversation history are not used to
influence subsequent requests.

## Reproducibility notes

API-based model outputs can change as hosted model implementations evolve.
Model identifiers, frozen thresholds, prompts, random seeds, runtime metadata,
and result files are therefore retained wherever available.

API credentials are not included in this repository.

## Data and results

Final analysis-ready outputs are provided for all three experiments. Each
experiment directory contains the code and the corresponding frozen inputs and
results required to audit the reported analyses.

## Citation

Please cite the accompanying paper if you use the framework, code, or released
experimental materials.
