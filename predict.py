import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier import clean_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--rule', required=True)
    parser.add_argument('--comment', required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    inputs = tokenizer(clean_text(args.rule), clean_text(args.comment), return_tensors='pt', truncation=True, max_length=(model.config.task_specific_params or {}).get('max_length', 256))
    with torch.inference_mode():
        probability = model(**inputs).logits.softmax(-1)[0, 1].item()
    print(json.dumps({'rule_violation_probability': probability, 'threshold': 0.5, 'prediction': int(probability >= 0.5)}))
