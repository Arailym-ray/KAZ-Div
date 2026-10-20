# KAZ-Div

**KAZ-Div: A Stateless Model-Agnostic Framework for Quality-Constrained Diversity in Kazakh Language Generation**

<img width="6800" height="4012" alt="kaz_div_framework_v2" src="https://github.com/user-attachments/assets/63ef29ba-7806-4aa4-9e40-2e79602c5237" />

**Supplementary materials:** [Zenodo DOI: 10.5281/zenodo.23504237](https://doi.org/10.5281/zenodo.23506385) (revised release dated **20 October 2026**, as reported by the authors). The revised supporting materials are identified as **Data S1–S4** and **Supplementary Tables S1–S20**.

This repository contains implementation code, frozen research inputs, task prompts, experimental configurations, saved results, statistical comparisons, and reproducibility records for the KAZ-Div study. The supplementary package on Zenodo is the referenced data deposit; some supplementary tables and human-evaluation materials are not mirrored at the repository root.

## Research scope

KAZ-Div is a stateless, model-agnostic *inference-time candidate-curation framework*. The experimental study covers three tasks:

1. **Kazakh Paraphrase Generation** — repeated-response lexical and embedding-based semantic diversity, source fidelity, and human-rated linguistic quality.
2. **Diverse Grounded Kazakh Question Generation** — question–answer generation conditioned on source passages, with automated structural and answer-span constraints.
3. **Synthetic Kazakh Intent Data Generation and Downstream Classification** — training-data diversity, automatic intent-consistency checks, and downstream classification performance.

The evaluated generators are **GPT-5.6 Sol**, **Claude Sonnet 4.6**, `issai/LLama-3.1-KazLLM-1.0-8B`, and `NousResearch/Meta-Llama-3.1-8B-Instruct`. The aim is to compare Baseline and KAZ-Div *within each model*, not to rank heterogeneous LLMs.

## Implemented KAZ-Div workflow

For each KAZ-Div request, a task-specific prompt requests a target of **12 candidates per generation call**, with up to **two candidate-generation calls** under the task's recovery procedure. The implemented processing sequence is:

1. **Parse and normalise** the generated candidates; remove normalised *exact duplicates* while preserving the earliest occurrence.
2. **Apply task-specific automated quality filtering**, including source-fidelity and length constraints for paraphrasing, question–answer/answer-span checks for grounded question generation, and intent-consistency constraints for synthetic utterances.
3. **Construct a semantic similarity graph** from BGE-M3 CLS-pooled, L2-normalised embeddings. Connected components define semantic near-duplicate clusters.
4. **Choose a representative per cluster** according to task-specific relevance/fidelity scores, with deterministic tie-breaking.
5. **Sample uniformly among cluster representatives** using a request-derived pseudorandom seed; this is random sampling with reproducible outcomes under fixed inputs and seed, not a deterministic best-cluster rule.

**Task-specific recovery matters:** Paraphrase and intent generation process each candidate-generation attempt independently; candidates from unsuccessful earlier attempts are not pooled with later attempts. Grounded question generation may incrementally accumulate exact-deduplicated question–answer candidates across calls. No previous *independent* request's candidates or outputs influence the current selection.

KAZ-Div does not fine-tune LLM weights and does not synthesize a new response during post-generation selection. Automated quality-pass scores are partly defined by the pipeline's own acceptance rules; they do **not** constitute independent human validation.

## Evaluations and available files

| Study | Repository path | Purpose |
| --- | --- | --- |
| **Kazakh Paraphrase Generation** | [`experiments/experiment_1_paraphrase/`](experiments/experiment_1_paraphrase/) | Five repeated outputs per source, lexical and semantic metrics, fidelity and quality analyses |
| **Diverse Grounded Kazakh Question Generation** | [`experiments/experiment_2_question_generation/`](experiments/experiment_2_question_generation/) | Source-grounded questions; automated answer-span checks; failure audit |
| **Synthetic Kazakh Intent Data Generation and Downstream Classification** | [`experiments/experiment_3_intent_classification/`](experiments/experiment_3_intent_classification/) | Synthetic intent training sets; TF-IDF/Logistic Regression; fixed held-out evaluation |
| **Stage-Wise Post-Hoc Ablation Analysis** | [`experiments/experiment_4_ablation/`](experiments/experiment_4_ablation/) | B0–B3 stages, same-pool comparisons, Supplementary Tables S14–S15 |
| **Additional Prompting and Decoding Baselines** | `experiments/Experiment 5 — Strong Diversity Baselines Compariso/` | Diversity-oriented prompting, higher temperature and taboo-word prompting (S16) |
| **Independent Encoder-Based Fidelity Validation** | `experiments/KAZ_Div_Experiment6_LaBSE_Fidelity_Results/` and `experiments/ KAZ_Div_Experiment6_e5_Fidelity_Results/` | LaBSE / multilingual-e5-base rescoring and encoder agreement |
| **Post-Hoc Parameter Sensitivity and Cluster-Chaining Analysis** | `experiments/KAZ_Div_Experiment7_Sensitivity_Results/` | Fidelity and duplicate thresholds, retrospective candidate availability and graph chaining |
| **Deduplication-Controlled Downstream Reassessment** | `experiments/KAZ_Div_Experiment8_Results/` | Unique-example matching, intent stratification and TF-IDF reassessment |
| **Multi-Seed Transformer-Based Downstream Classification Evaluation** | `experiments/KAZ_Div_Exp9/` | XLM-RoBERTa-base, group-aware validation, three training seeds (24 runs) |

These are the current **existing directory names**; some contain spaces or legacy naming. For definitive publication-level supplementary table numbering, consult the [Zenodo supporting-information record](https://doi.org/10.5281/zenodo.23504237).

## Interpretation and limitations

- **Stage-wise ablation:** B0 is conventional single-response generation, B1 random selection from reconstructed raw candidate pools, B2 random selection after quality filtering, and B3 the complete pipeline. B1–B3 share archived candidate-generation data, but not identical post-processing cost. Most of the increase over B0 occurs at B0–B1; B0 and B1 also differ in prompt and candidate-generation budget. Post-generation curation does **not** consistently improve lexical diversity compared with B1.
- **Statistical scope:** Planned paired tests cover B1–B0, B2–B1 and B3–B2 (21 tests per model). B3–B1 is a descriptive, not confirmatory, comparison.
- **Quality versus diversity:** Automated acceptance indicates consistency with frozen operational criteria, not independently demonstrated grammatical correctness or semantic entailment. Independent human and alternative-encoder checks provide complementary, imperfect evidence.
- **Compute:** Candidate generation and embedding-based curation introduce additional tokens, inference calls, latency, processing and, for proprietary models, API cost.
- **Parameter sensitivity:** Archived-pool truncation at smaller K is not the same as generating new candidate pools using K-specific prompts. BGE-M3 cosine thresholds should not be transferred directly to LaBSE or multilingual-e5-base.
- **Downstream classification:** Row-equalised corpora can have very different numbers of unique utterances. Unique-matched TF-IDF results use **five training-example sampling seeds**, while XLM-R analyses use **three classifier-training seeds**. These measure different sources of variability. Neither study independently regenerates the synthetic corpora; the transformer experiments are instance-matched, not unique-matched.
- **Test-set and annotation provenance:** Downstream classification uses a frozen held-out labelled set of 1,000 utterances (100 per intent). Its origin should be supported by source documentation, not inferred from filenames. Human evaluation must be interpreted using model-specific matched-source statistics alongside the ethics and consent records.

## Reproducibility and software

Root-level files include [`requirements.txt`](requirements.txt), [`CITATION.cff`](CITATION.cff), and [`KAZ-Div_MANIFEST_latest.csv`](KAZ-Div_MANIFEST_latest.csv). The accompanying replacement [`KAZ-Div_MANIFEST_latest.csv`](KAZ-Div_MANIFEST_latest.csv) lists **297 files** in the updated 20 October 2026 supplementary ZIP (excluding its own embedded manifest), with relative paths, byte sizes, SHA-256 hashes and source provenance. It corresponds to the available local updated archive and should be placed in the repository root with this README. The live Zenodo package should be checked against that archive before treating the checksum inventory as a verification of the public deposition.

Install the general Python dependencies with:

```bash
pip install -r requirements.txt
```

Some studies need additional experiment-specific libraries, pretrained model weights, GPU resources, and access to proprietary inference APIs. Hosted API responses may change over time even with frozen prompts and parameters. **No API credentials are included.** Some later experimental directories contain results and configuration manifests rather than complete self-contained execution scripts; consult each directory's files to assess rerun requirements.

## Supplementary materials

**[KAZ-Div supplementary release — Zenodo](https://doi.org/10.5281/zenodo.23504237)**. The revised package, dated **20 October 2026**, covers **Data S1–S4 and Tables S1–S20**, including the additional prompting/decoding baselines, independent encoder-based fidelity validation, parameter sensitivity and cluster-chaining analysis, deduplication-controlled downstream reassessment, multi-seed transformer classification, and model-specific human-evaluation statistics.

## Citation

Cite the [Zenodo record](https://doi.org/10.5281/zenodo.23504237) when using the supplementary data, and cite the KAZ-Div paper when its final publication metadata become available. The replacement [`CITATION.cff`](CITATION.cff) in this update package aligns the author list with the revised manuscript (Arailym Tleubayeva, Zhansaya Makhambetova, Altynbek Seitenov and Pinar Sarisaray Boluk). Verify the author list and publication metadata with all authors before creating a software release.
