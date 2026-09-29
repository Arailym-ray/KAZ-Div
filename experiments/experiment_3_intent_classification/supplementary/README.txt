KAZ-Div — Supplementary Materials for Synthetic Intent Data Diversity and Downstream Classification

Final generation status
=======================
Scheduled generation tasks: 4000
Successful synthetic generations: 3965
Terminal method-level failures: 35
Finalised tasks: 4000
Remaining tasks: 0

Package structure
=================
S1_Design_and_Frozen_Data: frozen configuration, signature, prompts, intent definitions, human test set, completion status.
S2_Generation_Outputs: final successful generations and terminal failures.
S3_Diversity_and_Quality: intrinsic diversity and quality summary.
S4_Downstream_Classification: equalised training corpus, classifier results, predictions, reports, and statistical tests.
S5_Confusion_Matrices: Baseline/KAZ-Div confusion matrices for the four providers using primary seed 20260922.

Notes
=====
- Baseline and KAZ-Div downstream training sets were equalised within each provider.
- Exact train-test overlaps were removed before classifier training.
- Downstream classifier: word + character TF-IDF with Logistic Regression.
- Five random seeds were used for descriptive downstream evaluation.
- McNemar/bootstrap/Holm statistics refer to the pre-specified primary seed 20260922.
- Intermediate checkpoints, resume archives, and transient runner errors are intentionally excluded.
