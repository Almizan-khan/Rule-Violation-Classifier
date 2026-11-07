import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/benchmark.json')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = DebertaV2Config(vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                            intermediate_size=64, max_position_embeddings=128, num_labels=2, problem_type='single_label_classification')
    model = DebertaV2ForSequenceClassification(config).eval()
    inputs = torch.randint(1, 128, (8, 64))
    durations = []
    with torch.inference_mode():
        for index in range(55):
            started = time.perf_counter()
            model(input_ids=inputs)
            if index >= 5:
                durations.append((time.perf_counter()-started)*1000)
    result = {'scope': 'tiny randomly initialized DeBERTa architecture on synthetic IDs; not pretrained DeBERTa-v3 performance or classification accuracy',
              'seed': 42, 'batch_size': 8, 'sequence_length': 64, 'warmup': 5, 'iterations': 50,
              'parameters': sum(p.numel() for p in model.parameters()), 'median_ms': statistics.median(durations),
              'p95_ms': sorted(durations)[47], 'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform()}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
