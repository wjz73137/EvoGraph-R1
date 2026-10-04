"""Offline guard checks for the one-image, one-GPU policy smoke test."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_evqa_gpu_smoke import idle_gpu, local_snapshot_valid, sanitized


class SmokeGuardTests(unittest.TestCase):
    def test_idle_selected_gpu_is_accepted_without_touching_other_gpus(self):
        with patch('scripts.run_evqa_gpu_smoke.subprocess.check_output', side_effect=[
                '0, GPU-A, 24500\n1, GPU-B, 22000\n', 'GPU-B, 123\n']):
            self.assertEqual(idle_gpu(0)['physical_gpu'], 0)

    def test_busy_selected_gpu_is_refused(self):
        with patch('scripts.run_evqa_gpu_smoke.subprocess.check_output', side_effect=[
                '0, GPU-A, 24500\n', 'GPU-A, 123\n']):
            with self.assertRaisesRegex(RuntimeError, 'compute process'):
                idle_gpu(0)

    def test_low_free_memory_is_refused(self):
        with patch('scripts.run_evqa_gpu_smoke.subprocess.check_output', side_effect=[
                '0, GPU-A, 19000\n', '']):
            with self.assertRaisesRegex(RuntimeError, '20000 MiB'):
                idle_gpu(0)

    def test_signed_query_and_url_credentials_are_not_logged(self):
        message = sanitized('error https://name:password@example.org/a?token=secret#fragment')
        self.assertEqual(message, 'error https://example.org/a')

    def test_partial_model_snapshot_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'config.json').write_text(json.dumps({'model_type': 'qwen2_5_vl'}))
            (path/'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'param': 'model-00001-of-00002.safetensors'}}))
            with self.assertRaisesRegex(RuntimeError, 'missing or empty'):
                local_snapshot_valid(path)


if __name__ == '__main__':
    unittest.main()
