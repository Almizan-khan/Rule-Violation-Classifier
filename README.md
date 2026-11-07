# Rule-conditioned comment classification

A DeBERTa training pipeline that estimates whether a comment violates a supplied rule, with group-aware splits, a TF-IDF baseline and explicit classification loss.

[![checks](https://github.com/Almizan-khan/Rule-Violation-Classifier/actions/workflows/checks.yml/badge.svg)](https://github.com/Almizan-khan/Rule-Violation-Classifier/actions/workflows/checks.yml)


## What this demonstrates

- Paired rule/comment tokenization with consistent preprocessing at training and inference.
- Two-class cross-entropy classification, replacing the ambiguous single-logit regression configuration in the original experiment.
- Duplicate-comment or unseen-rule validation, baseline comparison, per-rule error analysis and saved inference artifacts.

## Dataset contract

Training CSV columns: `body`, `rule`, `rule_violation` (integer 0 or 1). Optional competition test data uses `body`, `rule`, `row_id`. Download competition data through your own authorized Kaggle account; no private dataset or credentials are bundled.

Whitespace is normalized, but URLs and punctuation are preserved because they can be evidence for a rule violation. By default, identical normalized comments remain in one split. `--split rule` instead holds out entire rule strings to study generalization to unseen rules. A small dataset that cannot provide both classes in both splits is rejected.

## Train and predict

Use Python 3.11. A CUDA GPU is recommended for full DeBERTa training:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python classifier.py --train data/train.csv --baseline-only --output outputs/baseline
python rule-violation.py --train data/train.csv --test data/test.csv --output outputs/deberta
python predict.py --model outputs/deberta/model --rule "No promotional links" --comment "Read the community guidelines."
```

The initial model download requires internet access. The default is `microsoft/deberta-v3-base`, four epochs, batch size 8 and maximum sequence length 256. `--seed`, `--split`, `--epochs`, `--batch-size` and `--max-length` are configurable.

## Evaluation and artifacts

TF-IDF + logistic regression and the Transformer use the same split. Metrics include ROC-AUC, average precision, accuracy, macro-F1 and a confusion matrix at threshold 0.5. Single-class per-rule groups report undefined ranking metrics as null rather than inventing an AUC.

The output directory contains a reloadable model/tokenizer, `metrics.json`, misclassified rows in `errors.csv` and, when supplied a test CSV, an ID-aligned `submission.csv`. Reports include a dataset hash and training configuration. Checkpoints are selected by validation AUC, so these validation scores are not an independent final test estimate.

## Verification

```bash
python -m unittest discover -s tests -v
python benchmark.py --output reports/architecture-benchmark.json
```

Tests verify the actual cross-entropy loss and finite gradients, stable probability conversion, URL preservation and comment-group isolation. The CI benchmark runs a **tiny randomly initialized DeBERTa configuration on synthetic token IDs**. Its CPU latency is not the latency or quality of the pretrained base model.

The historical 0.91 validation / 0.84 leaderboard claims are retained in [the original README](docs/original-readme.md); they are not treated as reproduced measurements. No current task-quality score is claimed without the actual dataset and saved run artifacts.

## Recorded architecture check

[The executed CI report](reports/architecture-latency.json) records six passing tests and a median 1.03 ms / p95 1.34 ms forward pass for the **17,922-parameter random test model**, batch size 8 and sequence length 64. This is an implementation check, not a benchmark of the pretrained DeBERTa-v3-base model or a moderation-quality result.

## Limitations and next experiment

Near-duplicates can still cross a comment-based split. Rule-based splitting evaluates a different question and can produce a harder distribution shift. Evaluate both, inspect errors by rule and language, and tune thresholds only on validation data before a separate held-out test. A moderation score should support human review rather than define universal policy.

See [validation](VALIDATION.md) and [original experiment](notebooks/original_experiment.ipynb). Code is [MIT licensed](LICENSE); dataset and pretrained-model terms remain separate.

## Reference

[Hugging Face DeBERTa-v2/v3 model documentation](https://huggingface.co/docs/transformers/en/model_doc/deberta-v2)
