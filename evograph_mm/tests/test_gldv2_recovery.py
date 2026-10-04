"""Offline identity checks: never recover a merely similar landmark photograph."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.recover_gldv2_missing import (
    files_with_sha1, identity_evidence, image_query, sha1_hex, verified_sources, VerifiedDownloader,
)
from scripts.download_gldv2_thumbnails import Downloader
import requests


class RecoveryTests(unittest.TestCase):
    def test_base36_and_hex_sha1_have_identical_normal_form(self):
        self.assertEqual(sha1_hex("1"), "0" * 39 + "1")
        self.assertEqual(sha1_hex("0" * 39 + "A"), "0" * 39 + "a")
        with self.assertRaises(ValueError):
            sha1_hex("not-a-hash")

    def test_move_log_ignores_deleted_redirect_and_previous_unrelated_file(self):
        history = {"events": [
            {"type": "delete", "action": "delete", "logpage": 20},
            {"type": "move", "action": "move", "logpage": 10, "logid": 9,
             "params": {"target_ns": 6, "target_title": "File:Renamed.jpg"}},
            {"type": "upload", "logpage": 10, "params": {"img_sha1": "1"}},
            {"type": "upload", "logpage": 5, "params": {"img_sha1": "2"}},
        ]}
        proof = identity_evidence(history)
        self.assertEqual(proof["expected_sha1s"], [sha1_hex("1")])
        self.assertEqual(proof["move_target"], "File:Renamed.jpg")

    def test_same_filename_different_sha1_is_not_accepted(self):
        pages = {"1": {"title": "File:Same.jpg", "imageinfo": [
            {"sha1": "2", "thumburl": "https://upload.wikimedia.org/thumb.jpg"}]}}
        self.assertEqual(verified_sources(pages, {"id": {"expected_sha1s": [sha1_hex("1")]}},
                                          "en.wikipedia.org"), {})

    def test_exact_sha1_and_official_thumbnail_are_accepted(self):
        pages = {"1": {"title": "File:Renamed.jpg", "imageinfo": [
            {"sha1": "1", "thumburl": "https://upload.wikimedia.org/thumb.jpg"}]}}
        result = verified_sources(pages, {"id": {"expected_sha1s": [sha1_hex("1")]}},
                                  "commons.wikimedia.org")
        self.assertEqual(result["id"]["sha1"], sha1_hex("1"))

    def test_large_original_is_never_downloaded_without_a_thumbnail(self):
        pages = {"1": {"title": "File:Large.jpg", "imageinfo": [
            {"sha1": "1", "width": 6000, "url": "https://upload.wikimedia.org/original.jpg"}]}}
        self.assertEqual(verified_sources(pages, {"id": {"expected_sha1s": [sha1_hex("1")]}},
                                          "en.wikipedia.org"), {})

    def test_missing_original_sha1_cannot_be_assumed_identical(self):
        pages = {"1": {"title": "File:Same.jpg", "imageinfo": [
            {"sha1": "1", "thumburl": "https://upload.wikimedia.org/thumb.jpg"}]}}
        self.assertEqual(verified_sources(pages, {"id": {"expected_sha1s": []}},
                                          "en.wikipedia.org"), {})

    def test_metadata_proxy_never_changes_direct_image_session(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def json(self):
                return {"query": {"pages": {}}}
        with tempfile.TemporaryDirectory() as folder:
            d = VerifiedDownloader(Path(folder), Path(folder))
            direct = d.session
            captured = []
            def fake_get(*args, **kwargs):
                captured.append((d.session, dict(d.session.proxies), d.session.trust_env))
                self.assertEqual(kwargs['connection_attempts'], 1)
                return Response()
            with patch.object(d, "get", side_effect=fake_get):
                image_query(d, "en.wikipedia.org", ["File:A.jpg"])
            self.assertIs(d.session, direct)
            self.assertFalse(d.session.proxies)
            self.assertFalse(captured[0][2])
            self.assertEqual(captured[0][1]['https'], 'http://127.0.0.1:7890')
            d.session.close()

    def test_one_attempt_metadata_timeout_does_not_sleep_for_21_minutes(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Downloader(Path(folder), Path(folder))
            with patch.object(d.session, "get", side_effect=requests.ConnectTimeout), \
                    patch.object(d, "wait") as wait, \
                    patch("scripts.download_gldv2_thumbnails.shutil_disk_free", return_value=10**12):
                with self.assertRaises(requests.ConnectTimeout):
                    d.get('https://commons.wikimedia.org/w/api.php', 'metadata', connection_attempts=1)
                self.assertEqual(d.session.get.call_count, 1)
                self.assertFalse(any(call.args[-1] == 'direct_connection_retry' for call in wait.call_args_list))
            d.session.close()

    def test_hash_lookup_verifies_returned_hash_before_accepting_a_new_title(self):
        with patch('scripts.recover_gldv2_missing.api_query', return_value={
                'query': {'allimages': [
                    {'name': 'Translated.jpg', 'sha1': '1'},
                    {'name': 'Wrong.jpg', 'sha1': '2'},
                ]}}):
            self.assertEqual(files_with_sha1(None, 'uk.wikipedia.org', sha1_hex('1')),
                             ['File:Translated.jpg'])

    def test_hash_lookup_accepts_canonical_title_without_a_name_field(self):
        with patch('scripts.recover_gldv2_missing.api_query', return_value={
                'query': {'allimages': [{'title': 'File:Translated.jpg', 'sha1': '1'}]}}):
            self.assertEqual(files_with_sha1(None, 'commons.wikimedia.org', sha1_hex('1')),
                             ['File:Translated.jpg'])


if __name__ == "__main__":
    unittest.main()
