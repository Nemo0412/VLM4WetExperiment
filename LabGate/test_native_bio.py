import json
import unittest
from pathlib import Path

import numpy as np

import pipeline
from pipeline import LabGate
from prepare_egoproactive_bio_labgate import build_records, target_type
from prompts import judger_prompt


DATASET = Path("/home/gz2522/bio-dataset/EgoProactive-Bio")


class FakeModel:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, frames, prompt):
        self.calls.append((frames, prompt))
        return self.reply, 0.1


class NativeBioTests(unittest.TestCase):
    def test_mapping_preserves_original_labels_without_model_leakage(self):
        rows = build_records(DATASET)
        self.assertEqual(len(rows), 25)
        counts = {label: sum(row["gt_type"] == label for row in rows)
                  for label in ["none", "assistant", "safety", "action_error"]}
        self.assertEqual(counts, {"none": 11, "assistant": 8, "safety": 2, "action_error": 4})
        self.assertEqual(sum(row["gt_fire"] for row in rows), 14)
        model_fields = {"video", "frame_indices", "protocol", "asr_text"}
        for row in rows:
            self.assertEqual(row["asr_text"], "")
            model_payload = {key: row[key] for key in model_fields}
            self.assertNotIn(row["task_for_audit_only"], json.dumps(model_payload))
            if row["reference_guidance"]:
                self.assertNotIn(row["reference_guidance"], json.dumps(model_payload))

    def test_pair_mapping(self):
        def item(phase, pair, answer="$interrupt$x"):
            return {"decision_id": "d", "phase": phase, "pair_id": pair, "answer": answer}
        self.assertEqual(target_type(item("error", "uv_on")), ("safety", "safety"))
        self.assertEqual(target_type(item("error", "no_gloves")), ("safety", "safety"))
        self.assertEqual(target_type(item("error", "flask_dry")), ("action_error", "action_error"))
        self.assertEqual(target_type(item("protocol", None)), ("assistant", "next_step"))
        self.assertEqual(target_type(item("recovery", "x", "$silent$")), ("none", "none"))

    def test_native_prompt_is_minimally_extended(self):
        original = judger_prompt("P", "")
        proactive = judger_prompt("P", "", proactive_next_step=True)
        self.assertNotIn("next_step", original)
        self.assertIn("next_step", proactive)
        self.assertEqual(proactive.count("next_step"), 3)
        self.assertNotIn("JSON", proactive)
        self.assertNotIn("PREVIOUS DIALOG", proactive)

    def test_native_pipeline_passes_reason_and_only_calls_on_yes(self):
        old = pipeline.reproject_keep
        pipeline.reproject_keep = lambda frames, tau: (frames, list(range(len(frames))))
        try:
            for reply, fired in [("YES. reason=safety", True), ("YES. reason=next_step", True), ("NO.", False)]:
                with self.subTest(reply=reply):
                    expert = FakeModel("TYPE: ASSISTANT\nMSG: Continue.")
                    result = LabGate(FakeModel(reply), expert, proactive_next_step=True).run_frames(
                        np.zeros((2, 56, 56, 3), dtype=np.uint8), "P"
                    )
                    self.assertEqual(result.fired, fired)
                    self.assertEqual(len(expert.calls), int(fired))
                    if fired:
                        self.assertIn(result.judger_reason, result.expert_prompt)
        finally:
            pipeline.reproject_keep = old


if __name__ == "__main__":
    unittest.main()
