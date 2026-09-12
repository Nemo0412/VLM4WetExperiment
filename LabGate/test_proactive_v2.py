import json
import unittest

from proactive_v2 import clean_history, historical_indices, parse_structured_gate, regular_window, run_step, update_state
from prompts import expert_prompt, parse_judger


class Fake:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate_window(self, visual, system, text, history):
        self.calls.append((visual, system, text, history))
        return self.reply, .1


class Tests(unittest.TestCase):
    def test_reason_and_evidence_reach_expert_only_when_fired(self):
        visual = {'fps': 2, 'timestamps_sec': [1, 1.5], 'history_timestamps_sec': [], 'interval_sec': [0, 2]}
        for label in ['interrupt', 'silent']:
            reason = 'safety_warning' if label == 'interrupt' else 'none'
            payload = {'decision': label, 'reason_type': reason, 'observation': 'Bare hand visible',
                       'current_step': 'cabinet work', 'step_status': 'in_progress'}
            expert = Fake('Check your gloves.')
            result = run_step(Fake(json.dumps(payload)), expert, visual, 'task',
                              [{'role': 'user', 'text': 'task'}], [])
            self.assertEqual(len(expert.calls), int(label == 'interrupt'))
            if label == 'interrupt':
                for expected in ['safety_warning', 'Bare hand visible', 'cabinet work', 'unverified']:
                    self.assertIn(expected, expert.calls[0][2])
            state = []
            update_state(state, result, 2)
            self.assertEqual(state[0]['observation'], 'Bare hand visible')
            self.assertNotIn('decision', state[0])

    def test_sampler_regular_causal_even_and_history_separate(self):
        for start, end in [(0, 3), (33.567, 41.567), (245.867, 253.867), (717.167, 725.167)]:
            indices, fps = regular_window(start, end, 30, 21780)
            self.assertEqual(len(indices) % 2, 0)
            self.assertGreaterEqual(len(indices), 6)
            self.assertEqual(len(set(b-a for a,b in zip(indices, indices[1:]))), 1)
            self.assertAlmostEqual(30/(indices[1]-indices[0]), fps)
            self.assertTrue(all(start <= i/30 < end for i in indices))
            self.assertLess(end-indices[-1]/30, 1/30+1e-6)
            self.assertTrue(all(i/30 < start for i in historical_indices(start,30)))

    def test_history_tags_removed_and_roles_preserved(self):
        hist = [{'role':'user','text':'task'}] + [{'role':'assistant','text':f'$interrupt$Step {i}'} for i in range(7)]
        cleaned = clean_history(hist)
        self.assertEqual(len(cleaned), 5)
        self.assertNotIn('$interrupt$', str(cleaned))
        self.assertEqual(cleaned[1], {'role':'assistant','text':'Step 3'})

    def test_invalid_output_is_not_a_positive_decision(self):
        self.assertEqual(parse_structured_gate('Maybe YES')['decision'], 'invalid')
        self.assertEqual(parse_structured_gate('YES')['decision'], 'interrupt')
        self.assertFalse(parse_structured_gate('YES')['schema_valid'])
        self.assertEqual(parse_judger('Yesterday was fine'), (False, None))
        self.assertEqual(parse_judger('NO, but YES later'), (False, None))

    def test_ambiguous_json_decision_never_calls_expert(self):
        visual = {'fps': 2, 'timestamps_sec': [0, .5], 'history_timestamps_sec': [], 'interval_sec': [0, 1]}
        for decision in ['interrupt|silent', 'interrupt or silent', 'yes/no', ['interrupt'], None]:
            with self.subTest(decision=decision):
                raw = json.dumps({'decision': decision, 'reason_type': 'safety_warning', 'observation': 'Hand visible',
                                  'current_step': 'unknown', 'step_status': 'unknown'})
                expert = Fake('unused')
                result = run_step(Fake(raw), expert, visual, 'task', [], [])
                self.assertEqual(result['pred_label'], 'invalid')
                self.assertFalse(result['trigger']['schema_valid'])
                self.assertEqual(expert.calls, [])
        self.assertEqual(parse_structured_gate('interrupt|silent')['decision'], 'invalid')
        self.assertFalse(parse_structured_gate(json.dumps({'decision':'interrupt','step_status':[]}))['schema_valid'])

    def test_original_expert_prompt_receives_category(self):
        self.assertIn('"reason_type": "safety"', expert_prompt('protocol', '', 'safety'))
        self.assertNotIn('SMALL VLM TRIGGER', expert_prompt('protocol', ''))


if __name__ == '__main__':
    unittest.main()
