"""CPU-only regression checks for gate routing, leakage boundaries and scoring."""

import unittest

from egoproactive_bio import context_text, parse_gate, run_gated, sample_indices, summarize


class FakeModel:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, frames, prompt):
        self.calls.append((frames, prompt))
        return self.reply, 0.1


class BioTests(unittest.TestCase):
    def test_only_positive_decisions_call_expert(self):
        for raw in ('$interrupt$', 'YES. reason=safety', 'interrupt', 'interupt',
                    '$silent$', 'NO.', 'silent', 'Perhaps YES', ''):
            with self.subTest(raw=raw):
                expert = FakeModel('Check the protocol.')
                result = run_gated(FakeModel(raw), expert, 'frames', 'context')
                expected = raw in ('$interrupt$', 'YES. reason=safety', 'interrupt', 'interupt')
                self.assertEqual(len(expert.calls), int(expected))
                self.assertEqual(result['fired'], expected)
                if not expected:
                    self.assertEqual(result['guidance'], '')
                    self.assertEqual(result['latency_expert_generate_s'], 0)
        self.assertEqual(parse_gate('NO, but YES for another step'), 'silent')
        self.assertEqual(parse_gate('yesterday'), 'invalid')

    def test_causal_sampling_covers_window(self):
        for start, end in ((0, 3), (15, 18), (33.567, 41.567), (717.167, 725.167)):
            indices = sample_indices(start, end, 30, 21780)
            self.assertEqual(indices, sorted(set(indices)))
            self.assertLess(indices[-1] / 30, end)
            self.assertLess(end - indices[-1] / 30, 1 / 30 + 1e-6)
            window = [i / 30 for i in indices if i / 30 >= start]
            self.assertGreaterEqual(len(window), 8)
            self.assertLess(window[0] - start, 1 / 30 + 1e-6)
        with self.assertRaises(ValueError):
            sample_indices(3, 2, 30, 100)

    def test_context_has_only_explicit_history_and_static_sop(self):
        text = context_text('Cell passaging', [{'role': 'user', 'text': 'Cell passaging'}], [0, 2.9], 3)
        for forbidden in ('phase', 'pair_id', 'Video1', 'Step 3 error', 'source_decision_interval'):
            self.assertNotIn(forbidden, text)

    def test_metrics_count_invalid_as_error(self):
        rows = []
        for gold, pred in [('interrupt', 'interrupt'), ('interrupt', 'silent'),
                           ('silent', 'interrupt'), ('silent', 'silent'), ('silent', 'invalid')]:
            rows.append(dict(gt_label=gold, pred_label=pred, fired=pred == 'interrupt',
                             guidance_nonempty=pred == 'interrupt', phase='protocol',
                             n_frames_in=16, n_frames_kept=8, latency_wall_s=1,
                             latency_judger_generate_s=.1, latency_expert_generate_s=.2))
        result = summarize(rows)
        self.assertEqual(result['accuracy'], 2 / 5)
        self.assertEqual(result['invalid_outputs'], 1)
        self.assertEqual(result['expert_calls'], 2)
        self.assertAlmostEqual(result['per_class']['interrupt']['f1'], .5)
        self.assertAlmostEqual(result['per_class']['silent']['f1'], .4)
        self.assertAlmostEqual(result['macro_f1'], .45)
        self.assertIsNone(result['guidance_semantic_accuracy'])


if __name__ == '__main__':
    unittest.main()
