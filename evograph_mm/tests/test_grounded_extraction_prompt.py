"""Checks for the opt-in grounded local-Qwen extraction prompt."""

import unittest
from unittest.mock import patch

from graphr1.prompt import PROMPTS
from graphr1.prompt_grounded import (
    GROUNDED_ENTITY_EXTRACTION,
    GROUNDED_PROMPT_VERSION,
    GROUNDED_REVIEW_PROMPT,
    activate_grounded_prompts,
)
from scripts import run_evqa_native_graph_grounded as grounded_runner
from scripts.run_evqa_native_graph_smoke import validate_extraction_output


class GroundedExtractionPromptTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(PROMPTS)

    def tearDown(self):
        PROMPTS.clear()
        PROMPTS.update(self.original)

    def test_prompt_forbids_external_knowledge_and_adjacent_event_merging(self):
        self.assertIn("Use ONLY information explicitly stated", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("Do not combine facts from adjacent sentences", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("Do not duplicate semantically equivalent", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("Do not create a separate entity for an alias", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("do not begin a record with a pronoun", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("Do not output only", GROUNDED_ENTITY_EXTRACTION)
        self.assertIn("COMPLETE REPLACEMENT", GROUNDED_REVIEW_PROMPT)
        self.assertIn("who or what was seized", GROUNDED_REVIEW_PROMPT)

    def test_activation_uses_qwen_safe_delimiters_without_changing_baseline_file(self):
        activate_grounded_prompts()
        self.assertEqual(PROMPTS["DEFAULT_TUPLE_DELIMITER"], "@@")
        self.assertEqual(PROMPTS["DEFAULT_RECORD_DELIMITER"], "##")
        self.assertNotIn("<|>", PROMPTS["entity_extraction"])
        rendered = PROMPTS["entity_extraction"].format(
            language="English",
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            examples=PROMPTS["entity_extraction_examples"][0].format(
                tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
                record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
                completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            ),
            input_text="A source fact.",
        )
        self.assertIn('("hyper-relation"@@', rendered)
        self.assertTrue(GROUNDED_PROMPT_VERSION)

    def test_runner_selects_model_output_and_token_limit_explicitly(self):
        model = grounded_runner.smoke.ROOT.parent / "models" / "Qwen2.5-VL-7B-Instruct"
        with patch.object(grounded_runner.smoke, "main", return_value=0) as run:
            result = grounded_runner.main(
                ["--model-path", str(model), "--output-tag", "evqa_native_graph_grounded_7b",
                 "--max-new-tokens", "2048", "--gpu", "0"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(grounded_runner.smoke.MODEL, model.resolve())
        self.assertEqual(grounded_runner.smoke.MAX_NEW_TOKENS, 2048)
        self.assertEqual(grounded_runner.smoke.PROMPT_VERSION, GROUNDED_PROMPT_VERSION)
        self.assertEqual(grounded_runner.smoke.PREFIX, "evqa_native_graph_grounded_7b")
        run.assert_called_once_with(["--gpu", "0"])

    def test_runner_enables_review_pass_explicitly(self):
        model = grounded_runner.smoke.ROOT.parent / "models" / "Qwen2.5-VL-7B-Instruct"
        with patch.object(grounded_runner.smoke, "main", return_value=0):
            grounded_runner.main(
                ["--model-path", str(model), "--output-tag", "evqa_native_graph_grounded_7b_v3",
                 "--max-new-tokens", "2048", "--review-pass", "--gpu", "0"]
            )
        self.assertTrue(grounded_runner.smoke.REVIEW_EXTRACTION)
        self.assertEqual(grounded_runner.smoke.REVIEW_PROMPT, GROUNDED_REVIEW_PROMPT)

    def test_terminator_only_generation_is_rejected_before_graph_insertion(self):
        with self.assertRaisesRegex(RuntimeError, "no parseable"):
            validate_extraction_output("<COMPLETE>")
        valid = '("hyper-relation"@@Fact.@@10)##\n("entity"@@Thing@@object@@Thing in fact.@@90)##'
        self.assertEqual(validate_extraction_output(valid), valid)


if __name__ == "__main__":
    unittest.main()
