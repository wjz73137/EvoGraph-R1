"""Offline checks: GME pooling, evidence provenance and label isolation."""

import unittest

from evograph_mm.kb.indexing.gme_compat import gme_prompt, last_token_pool
from scripts.run_evqa_retrieval_smoke import (answer_prompt, chunks, diagnostic_match,
                                            page_passages, query_payload)
from scripts.run_evqa_native_graph_smoke import graph_bundle, normalize_record_delimiters


class RetrievalSmokeTests(unittest.TestCase):
    def test_provider_prompt_is_preserved(self):
        self.assertEqual(gme_prompt('question', True, 'instruction'),
                         '<|im_start|>system\ninstruction<|im_end|>\n'
                         '<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>question'
                         '<|im_end|>\n<|im_start|>assistant\n<|endoftext|>')

    def test_pooling_uses_actual_last_unmasked_token(self):
        import torch
        hidden = torch.arange(16).reshape(2, 4, 2)
        pooled = last_token_pool(hidden, torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]]))
        self.assertTrue(torch.equal(pooled, torch.tensor([[2, 3], [14, 15]])))
        with self.assertRaises(ValueError):
            last_token_pool(hidden, torch.zeros((2, 4), dtype=torch.int64))

    def test_chunk_coverage_and_determinism(self):
        text = 'some genuine encyclopedia text. ' * 120
        result = list(chunks(text, 120, 15))
        self.assertEqual(result, list(chunks(text, 120, 15)))
        self.assertEqual(result[-1][1], len(text))
        self.assertTrue(all(end-start <= 120 for start, end, _ in result))
        self.assertTrue(all(result[i+1][0] < result[i][1] for i in range(len(result)-1)))

    def test_passages_keep_url_section_correspondence_without_qa(self):
        pages = {'url': {'url': 'url', 'title': 'Castle', 'section_titles': ['History'],
                         'section_texts': ['Built in 1400.'], 'answer': 'DO NOT COPY'}}
        doc = page_passages(pages)[0]
        self.assertEqual(doc['text'], 'Built in 1400.')
        self.assertEqual(doc['section_id'], 0)
        self.assertNotIn('DO NOT COPY', str(doc))

    def test_query_allowlist_excludes_gold_fields(self):
        row = {'question': 'When?', 'answer': 'SECRET_REFERENCE',
               'wikipedia_title': 'SECRET_ENTITY', 'wikipedia_url': 'SECRET_URL'}
        self.assertEqual(query_payload(row, '/local/image.jpg'),
                         {'text': 'When?', 'image': '/local/image.jpg'})
        self.assertNotIn('SECRET', answer_prompt(row['question']))

    def test_phrase_match_supports_official_multianswer_syntax_not_substrings(self):
        self.assertTrue(diagnostic_match('A and B.', 'a&&b|c'))
        self.assertFalse(diagnostic_match('A', 'a&&b'))
        self.assertFalse(diagnostic_match('unknown', 'known'))
        self.assertFalse(diagnostic_match('15260', '1526'))

    def test_native_graph_uses_only_real_kb_text_and_safe_metadata(self):
        rows = [{'wikipedia_url': 'url', 'dataset_image_ids': 'img',
                 'question': 'SECRET_QUESTION', 'answer': 'SECRET_REFERENCE',
                 'evidence': 'SECRET_GOLD_EVIDENCE', 'wikipedia_title': 'SECRET_LABEL'}]
        pages = {'url': {'title': 'Official Castle', 'section_titles': ['History'],
                         'section_texts': ['Built in 1400.']}}
        bundle = graph_bundle(rows, pages, {'img': '/local/image.jpg'}, count=1)
        self.assertNotIn('SECRET', str(bundle))
        self.assertIn('Built in 1400.', bundle.text_documents[0]['contents'])
        self.assertEqual(bundle.text_documents[0]['source_metadata']['wikipedia_title'], 'Official Castle')

    def test_format_repair_does_not_invent_records_or_change_facts(self):
        raw = '("hyper-relation"<|>"Built in 1400."|>8)##\n'
        fixed, count = normalize_record_delimiters(raw)
        self.assertEqual(fixed, '("hyper-relation"<|>"Built in 1400."<|>8)##\n')
        self.assertEqual(count, 1)
        self.assertEqual(normalize_record_delimiters(fixed), (fixed, 0))
        self.assertEqual(normalize_record_delimiters('Unknown |>8)\n'), ('Unknown |>8)\n', 0))


if __name__ == '__main__':
    unittest.main()
