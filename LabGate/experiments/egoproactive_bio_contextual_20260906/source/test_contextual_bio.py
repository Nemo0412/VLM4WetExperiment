import unittest
from pathlib import Path

import numpy as np

from models import build_messages
from prepare_egoproactive_bio_contextual import (
    build_contextual_records,
    normalize_history,
    official_history,
)
from prompts import expert_prompt, judger_prompt


DATASET = Path("/home/gz2522/bio-dataset/EgoProactive-Bio")


class ContextualBioTests(unittest.TestCase):
    def test_history_and_current_step_are_model_context(self):
        gate, expert = build_contextual_records(DATASET)
        self.assertEqual((len(gate), len(expert)), (25, 14))
        self.assertEqual(gate[2]["current_step"], "Step 3 error — Do not open the biosafety cabinet while the UV light is on")
        self.assertTrue(gate[2]["history_official_raw"])
        self.assertTrue(gate[2]["history_official_clean"])
        self.assertTrue(any("$interrupt$" in turn["content"] for turn in gate[2]["history_official_raw"]))
        self.assertFalse(any("$interrupt$" in turn["content"] for turn in gate[2]["history_official_clean"]))

    def test_current_step_reaches_both_prompts_and_expert_trigger(self):
        step = "Step 8 — Discard the PBS rinse"
        small = judger_prompt("P", "", proactive_next_step=True, current_step=step)
        large = expert_prompt("P", "", "next_step", current_step=step)
        self.assertIn(step, small)
        self.assertGreaterEqual(large.count(step), 2)
        self.assertIn('"current_step": "' + step, large)
        self.assertIn("reason=user_query is forbidden", small)
        self.assertIn("not user speech or a request", small)
        self.assertIn("Do not repeat an earlier instruction", small)
        self.assertIn("recovery step being performed correctly is NO", small)

    def test_history_cleaning_keeps_roles_and_spoken_content(self):
        raw = [{"role": "user", "text": "Cell passaging"},
               {"role": "assistant", "text": "$interrupt$Add PBS now."}]
        clean = normalize_history(raw, strip_control_tokens=True)
        self.assertEqual(clean, [{"role": "user", "content": "Cell passaging"},
                                 {"role": "assistant", "content": "Add PBS now."}])

    def test_qwen_messages_preserve_history_before_current_video(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        history = [{"role": "user", "content": "Cell passaging"},
                   {"role": "assistant", "content": "Add PBS now."}]
        messages = build_messages(frames, "CURRENT TASK / STEP: discard PBS", history)
        self.assertEqual(messages[:2], history)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIs(messages[-1]["content"][0]["video"], frames)
        self.assertIn("discard PBS", messages[-1]["content"][1]["text"])

    def test_official_history_keeps_query_and_latest_four_turns(self):
        source = [{"role": "user", "text": "Cell passaging"}] + [
            {"role": "assistant", "text": f"$interrupt$step {i}"}
            for i in range(7)
        ]
        history = official_history(source, strip_control_tokens=False)
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["content"], "Cell passaging")
        self.assertEqual([x["content"] for x in history[1:]], [
            "$interrupt$step 3", "$interrupt$step 4",
            "$interrupt$step 5", "$interrupt$step 6",
        ])


if __name__ == "__main__":
    unittest.main()
