"""Offline regression tests for no-clobber downloads and fixed candidate subsets."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from evograph_mm.kb.gldv2_subset import (
    build_candidate_subset, image_info, publish_image,
)
from scripts.download_gldv2_thumbnails import (
    Downloader, HTTPFailure, SafeStop, imageinfo_url, retry_after_seconds, safe_url,
    limit_image_width, redirect_url,
)


class ThumbnailTests(unittest.TestCase):
    def test_retry_after_seconds_and_http_date(self):
        self.assertEqual(retry_after_seconds("600"), 600)
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        self.assertEqual(retry_after_seconds("Thu, 01 Jan 2026 00:01:00 GMT", epoch), 60)
        self.assertIsNone(retry_after_seconds("invalid"))

    def test_redaction_and_encoded_file_title(self):
        self.assertEqual(safe_url("https://user:secret@host/file?token=secret"),
                         "https://host/file")
        self.assertIn("titles=File%3APyrgos_%28Myrtos%29.jpg",
                      imageinfo_url("https://upload.wikimedia.org/commons/a/Pyrgos_%28Myrtos%29.jpg"))
        self.assertIn("iiurlwidth=1280", imageinfo_url("https://upload.wikimedia.org/a.jpg"))
        self.assertTrue(redirect_url("https://upload.wikimedia.org/a.jpg").endswith("width=1280"))

    def test_new_image_width_is_capped_at_1280_without_upscaling(self):
        with tempfile.TemporaryDirectory() as folder:
            temp, target = Path(folder)/"temp.jpg", Path(folder)/"target.jpg"
            Image.new("RGB", (2560, 1600)).save(temp)
            limit_image_width(temp, target)
            self.assertEqual((image_info(temp)["width"], image_info(temp)["height"]), (1280, 800))
            Image.new("RGB", (640, 400)).save(temp)
            before = temp.read_bytes()
            limit_image_width(temp, target)
            self.assertEqual(before, temp.read_bytes())

    def test_atomic_rename_does_not_replace_existing_image(self):
        with tempfile.TemporaryDirectory() as folder:
            target, temp = Path(folder)/"valid.jpg", Path(folder)/"temp.jpg"
            Image.new("RGB", (8, 8), "red").save(target)
            before = target.read_bytes()
            Image.new("RGB", (8, 8), "blue").save(temp)
            with self.assertRaises(FileExistsError):
                publish_image(temp, target)
            self.assertEqual(before, target.read_bytes())
            self.assertTrue(temp.exists())

    def test_html_payload_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"image.jpg"
            path.write_text("<html>429 error</html>")
            self.assertIsNone(image_info(path))

    def test_shared_server_stop_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            downloader = Downloader(root, root)
            self.assertFalse(downloader.session.trust_env)
            with patch("scripts.download_gldv2_thumbnails.shutil_disk_free", return_value=1):
                with self.assertRaisesRegex(SafeStop, "100GB"):
                    downloader.guard()
            with patch("scripts.download_gldv2_thumbnails.shutil_disk_free", return_value=10**12):
                downloader.consecutive_429 = 20
                with self.assertRaisesRegex(SafeStop, "20_consecutive"):
                    downloader.guard()
            downloader.session.close()

    def test_fixed_subset_keeps_order_split_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = root/"datasets_mm/E-VQA/raw/images/google_landmarks_v2/selected"
            selected.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(selected/"a.jpg")
            Image.new("RGB", (8, 8)).save(selected/"b.jpg")
            candidates = []
            for split, image_id, index in [("train", "b", 12), ("train", "a", 7),
                                            ("test", "b", 22)]:
                candidates.append({"split": split, "image_id": image_id, "row_index": index,
                                   "url": f"https://upload.wikimedia.org/{image_id}.jpg",
                                   "target_path": str(selected/f"{image_id}.jpg"),
                                   "qa_row": {"dataset_image_ids": image_id,
                                              "question": "Q", "answer": "A", "evidence": ""},
                                   "qa_sha256": "hash"})
            plan = {"name": "fixed", "seed": 0, "requested": {"train": 2, "test": 1},
                    "allow_buffer": False, "source_files": {}, "source_candidates": "original",
                    "source_sha256": "hash", "candidates": candidates,
                    "qa_fields": {s: list(candidates[0]["qa_row"]) for s in ("train", "test")}}
            report = build_candidate_subset(root, plan)
            self.assertTrue(report["complete"])
            subset = root/"datasets_mm/E-VQA/subsets/fixed"
            with (subset/"qa_train.csv").open() as f:
                self.assertEqual([r["dataset_image_ids"] for r in csv.DictReader(f)], ["b", "a"])
            manifest = json.loads((subset/"manifest.json").read_text())
            self.assertEqual([c["row_index"] for c in manifest["copied_images"]], [12, 7, 22])
            self.assertEqual(report["empty_evidence_preserved_from_raw"], 3)
            before = {p.name: p.read_bytes() for p in (subset/"images").iterdir()}
            self.assertTrue(build_candidate_subset(root, plan)["complete"])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (subset/"images").iterdir()})
            (selected/"a.jpg").write_text("bad")
            report = build_candidate_subset(root, plan)
            self.assertFalse(report["complete"])
            self.assertEqual(report["summary"]["complete_train"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (subset/"images").iterdir()})

    def test_retry_429_obeys_retry_after_and_spaces_requests(self):
        class Response:
            def __init__(self, status):
                self.status_code = status
                self.headers = {"Retry-After": "600"}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder:
            d = Downloader(Path(folder), Path(folder))
            with patch.object(d.session, "get", side_effect=[Response(429), Response(200)]), \
                    patch.object(d, "wait") as wait, \
                    patch("scripts.download_gldv2_thumbnails.shutil_disk_free", return_value=10**12):
                self.assertEqual(d.get("https://commons.wikimedia.org/w/api.php", "id").status_code, 200)
                self.assertIn(unittest.mock.call(600, "HTTP_429_retry_after"), wait.call_args_list)
                self.assertEqual(d.state["request_count"], 2)
            d.session.close()


if __name__ == "__main__":
    unittest.main()
