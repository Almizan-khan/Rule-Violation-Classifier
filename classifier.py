import argparse
import hashlib
import json
import platform
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

def clean_text(text):
    if text is None or pd.isna(text):
        return ''
    return re.sub(r'\s+', ' ', str(text)).strip()

def prepare_frame(frame, labels=True):
    required = {'body', 'rule'} | ({'rule_violation'} if labels else set())
    if not required.issubset(frame.columns):
        raise ValueError('Missing columns: '+', '.join(sorted(required-set(frame.columns))))
    result = frame.copy()
    for column in ['body', 'rule']:
        result[column] = result[column].map(clean_text)
        if (result[column] == '').any():
            raise ValueError('Empty body or rule values are not supported')
    if labels:
        if result['rule_violation'].isna().any() or not result['rule_violation'].isin([0, 1]).all():
            raise ValueError('rule_violation must contain binary labels 0 and 1')
        result['rule_violation'] = result['rule_violation'].astype(int)
    return result

def split_data(frame, mode='comment', seed=42):
    if mode not in {'comment', 'rule'}:
        raise ValueError('split mode must be comment or rule')
    groups = frame['body' if mode == 'comment' else 'rule']
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, groups=groups))
    training, validation = frame.iloc[train].copy(), frame.iloc[valid].copy()
    for part in [training, validation]:
        if part.rule_violation.nunique() != 2:
            raise ValueError('Both splits must contain both labels. Supply a larger or explicitly designed dataset.')
    return training, validation

def evaluate_probabilities(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError('Finite probabilities must align with labels')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be in [0, 1]')
    predictions = (probabilities >= 0.5).astype(int)
    return {'roc_auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            'average_precision': float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            'accuracy': float(accuracy_score(labels, predictions)), 'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist(), 'threshold': 0.5}

def probabilities_from_logits(logits):
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('Expected two logits per example')
    shifted = values-values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, 1]/exp.sum(axis=1)

def baseline(train, valid):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=30000), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train['rule']+' [SEP] '+train['body'], train.rule_violation)
    probabilities = model.predict_proba(valid['rule']+' [SEP] '+valid['body'])[:, 1]
    return evaluate_probabilities(valid.rule_violation, probabilities)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test')
    parser.add_argument('--model', default='microsoft/deberta-v3-base')
    parser.add_argument('--output', default='outputs/deberta')
    parser.add_argument('--split', choices=['comment', 'rule'], default='comment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    frame = prepare_frame(pd.read_csv(args.train))
    train, valid = split_data(frame, args.split, args.seed)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    report = {'config': vars(args), 'train_examples': len(train), 'validation_examples': len(valid),
              'dataset_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
              'baseline': baseline(train, valid), 'environment': {'python': platform.python_version(), 'platform': platform.platform()},
              'scope': 'validation metrics; checkpoint selection uses this split, so final claims require a separate held-out test set'}
    if not args.baseline_only:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        def encode(part, labeled=True):
            data = tokenizer(part.rule.tolist(), part.body.tolist(), truncation=True, max_length=args.max_length)
            if labeled:
                data['labels'] = part.rule_violation.tolist()
            return Dataset.from_dict(data)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, problem_type='single_label_classification')
        model.config.task_specific_params = {'max_length': args.max_length, 'threshold': 0.5}
        report['environment'].update({'torch': torch.__version__})
        def metrics(prediction):
            result = evaluate_probabilities(prediction.label_ids, probabilities_from_logits(prediction.predictions))
            return {k: v for k, v in result.items() if isinstance(v, (int, float))}
        configuration = TrainingArguments(output_dir=str(destination/'checkpoints'), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=2e-5, weight_decay=0.01, eval_strategy='epoch', save_strategy='epoch',
            load_best_model_at_end=True, metric_for_best_model='roc_auc', greater_is_better=True,
            save_total_limit=2, report_to='none', seed=args.seed, fp16=torch.cuda.is_available())
        trainer = Trainer(model=model, args=configuration, train_dataset=encode(train), eval_dataset=encode(valid),
                          processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
        trainer.train()
        trainer.save_model(str(destination/'model'))
        tokenizer.save_pretrained(str(destination/'model'))
        probabilities = probabilities_from_logits(trainer.predict(encode(valid)).predictions)
        report['transformer'] = evaluate_probabilities(valid.rule_violation, probabilities)
        report['per_rule'] = {rule: evaluate_probabilities(group.rule_violation, probabilities[positions])
                              for rule, positions in valid.reset_index(drop=True).groupby('rule').indices.items()
                              for group in [valid.iloc[positions]]}
        errors = valid[['body', 'rule', 'rule_violation']].copy()
        errors['probability'] = probabilities
        errors['predicted'] = (probabilities >= 0.5).astype(int)
        errors[errors.predicted != errors.rule_violation].to_csv(destination/'errors.csv', index=False)
        if args.test:
            test = prepare_frame(pd.read_csv(args.test), labels=False)
            if 'row_id' not in test:
                raise ValueError('Test data must contain row_id')
            test_probabilities = probabilities_from_logits(trainer.predict(encode(test, False)).predictions)
            pd.DataFrame({'row_id': test.row_id, 'rule_violation': test_probabilities}).to_csv(destination/'submission.csv', index=False)
    (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
