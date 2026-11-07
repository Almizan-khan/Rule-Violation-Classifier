import unittest
import numpy as np
import pandas as pd
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
from classifier import clean_text, prepare_frame, probabilities_from_logits, evaluate_probabilities, split_data

class ClassifierTests(unittest.TestCase):
    def test_urls_are_preserved_as_potential_rule_evidence(self):
        self.assertEqual(clean_text('  visit  https://example.com\n'), 'visit https://example.com')

    def test_two_class_loss_is_cross_entropy(self):
        torch.manual_seed(42)
        model = DebertaV2ForSequenceClassification(DebertaV2Config(vocab_size=32, hidden_size=24, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=48, max_position_embeddings=32, num_labels=2, problem_type='single_label_classification'))
        labels = torch.tensor([0, 1])
        output = model(input_ids=torch.randint(1, 32, (2, 8)), labels=labels)
        torch.testing.assert_close(output.loss, torch.nn.functional.cross_entropy(output.logits, labels))
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.classifier.weight.grad).all())

    def test_probability_conversion_is_stable(self):
        values = probabilities_from_logits(np.array([[1000., 1001.], [-1000., -999.]]))
        np.testing.assert_allclose(values, [0.73105858, 0.73105858])

    def test_binary_labels_and_required_columns_are_enforced(self):
        with self.assertRaises(ValueError):
            prepare_frame(pd.DataFrame({'body': ['text'], 'rule': ['rule'], 'rule_violation': [2]}))

    def test_duplicate_comments_do_not_cross_split(self):
        frame = pd.DataFrame({'body': [f'comment {i}' for i in range(30)]*2,
                              'rule': ['rule a']*30+['rule b']*30, 'rule_violation': [i%2 for i in range(30)]*2})
        train, valid = split_data(frame)
        self.assertFalse(set(train.body)&set(valid.body))

    def test_single_class_metric_has_no_fake_auc(self):
        self.assertIsNone(evaluate_probabilities([1, 1], [0.7, 0.8])['roc_auc'])
