# KAZ-Div

**KAZ-Div: A Stateless Model-Agnostic Framework for Quality-Constrained Diversity in Kazakh Language Generation**

This repository contains the reproducibility code, frozen experimental inputs,
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

Status in this repository snapshot: **code complete; recovered result file is
partial and is not the final paper result set**. See
`experiments/experiment_2_question_generation/STATUS.md`.

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
├── .gitignore
├── .gitattributes
├── MANIFEST.csv
└── experiments/
    ├── experiment_1_paraphrase/
    │   ├── code/
    │   ├── data/
    │   ├── results/
    │   └── supplementary/
    ├── experiment_2_question_generation/
    │   ├── code/
    │   ├── data/
    │   ├── results_partial/
    │   └── STATUS.md
    └── experiment_3_intent_classification/
        ├── code/
        ├── data/
        ├── results/
        └── supplementary/
```

## KAZ-Div procedure

For each request, KAZ-Div uses a temporary candidate pool and applies:

1. multi-candidate generation;
2. task-specific quality filtering;
3. normalized exact deduplication;
4. semantic near-duplicate clustering;
5. fidelity-aware representative selection within each cluster;
6. deterministic random selection among cluster representatives.

The framework is stateless across independent requests: previous generated
outputs, previous candidate pools, and conversation history are not used to
influence subsequent requests.

## Reproducibility notes

API-based model outputs can change as hosted model implementations evolve.
Model identifiers, frozen thresholds, prompts, random seeds, and result
signatures are therefore retained wherever available.

API credentials are **not** included in this repository. Supply credentials via
environment variables or a secrets manager, for example:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

For gated Hugging Face models, authenticate separately using the Hugging Face
CLI or notebook login.

## Installation

A typical Python environment can be prepared with:

```bash
pip install -r requirements.txt
```

GPU-backed execution is recommended for the locally deployed 8B models and
BGE-M3 embedding computation.

## Data and results

The repository retains final analysis-ready outputs for Experiments 1 and 3.
Experiment 2 currently contains the latest recovered partial main-generation
file only and is explicitly marked as incomplete.

## Citation

Please cite the accompanying paper if you use the framework, code, or released
experimental materials.

## License

A repository-wide licence has not yet been assigned. Select and add a licence
before public release.
