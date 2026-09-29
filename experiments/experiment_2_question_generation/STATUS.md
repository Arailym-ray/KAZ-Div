# Experiment 2 status

**Status: complete.**

The final full run contains 4,000 scheduled observations:

- 100 source passages
- 4 language models
- 2 generation methods: Baseline and KAZ-Div
- 5 repetitions per source/model/method

The final result set is stored in:

`results/experiment2_main_results.csv`

The repository also includes source-level metrics, primary summary metrics,
paired statistical tests, operational metrics, failure summaries, and
candidate-pool diagnostics.

Important interpretation note:
KazLLM Baseline had a substantially lower generation success rate because of
output-parsing failures. Consequently, only two KazLLM source passages formed
complete five-repetition Baseline/KAZ-Div pairs. Paired KazLLM diversity
statistics should therefore not be interpreted on the same evidential basis
as OpenAI, Anthropic, or Llama.
