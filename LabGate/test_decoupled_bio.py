import unittest
from pathlib import Path

from eval_egoproactive_bio_expert_oracle import score_semantics
from prepare_egoproactive_bio_decoupled import build_decoupled


DATASET = Path("/home/gz2522/bio-dataset/EgoProactive-Bio")


class DecoupledBioTests(unittest.TestCase):
    def test_datasets_are_disjoint_by_role(self):
        gate, expert = build_decoupled(DATASET)
        self.assertEqual(len(gate), 25)
        self.assertEqual(len(expert), 14)
        self.assertEqual(sum(row["gt_fire"] for row in gate), 14)
        self.assertFalse(any("reference_guidance" in row for row in gate))
        self.assertTrue(all(row["oracle_reason"] != "none" for row in expert))
        self.assertTrue(all(row["reference_guidance"] for row in expert))
        self.assertTrue(all(row["semantic_rubric"] for row in expert))

    def test_semantic_rubric_requires_every_concept_group(self):
        rubric = [["uv"], ["turn off", "switch off"]]
        self.assertTrue(score_semantics("Turn off the UV light.", rubric)["all_required_concepts"])
        partial = score_semantics("Check the UV light.", rubric)
        self.assertFalse(partial["all_required_concepts"])
        self.assertEqual(partial["concept_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()

