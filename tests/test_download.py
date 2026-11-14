"""Tests for verified model downloads and fallback routing.

Author: Tingyu Lin, Computer Vision Lab, TU Wien.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from persist_sbd import _download, api


class ReleaseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cache = self.root / "persist_sbd/github" / _download.DEFAULT_RELEASE
        self.payloads = {"config.json": b'{"threshold":0.5}\n', "model.safetensors": b"test weights only"}
        manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in self.payloads.items()}
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(_download, "DOWNLOAD_FILES", manifest))
        self.patches.enter_context(patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}))

    def response(self, request, **options):
        self.assertEqual(options, {"timeout": 30})
        name = request.full_url.rsplit("/", 1)[-1]
        return io.BytesIO(self.payloads[name])

    def cached(self):
        self.cache.mkdir(parents=True)
        for name, data in self.payloads.items():
            (self.cache / name).write_bytes(data)

    def assert_no_partial_files(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_first_download_then_verified_cache_reuse_without_requests(self):
        with patch.object(_download, "urlopen", side_effect=self.response) as download:
            files = _download.release_files(self.root)
            identities = [os.stat(path).st_ino for path in files]
            self.assertEqual(download.call_count, 2)
            self.assertEqual({call.args[0].full_url for call in download.call_args_list},
                             {_download.RELEASE_BASE_URL + "/" + name for name in self.payloads})
            download.reset_mock()
            self.assertEqual(_download.release_files(self.root), files)
            download.assert_not_called()
        self.assertEqual([os.stat(path).st_ino for path in files], identities)
        for path in files:
            self.assertEqual(Path(path).read_bytes(), self.payloads[Path(path).name])
        self.assert_no_partial_files()

    def test_bad_hash_is_rejected_without_publishing_the_file(self):
        original_response = self.response

        def wrong_bytes(request, **options):
            if request.full_url.endswith("model.safetensors"):
                return io.BytesIO(b"!" * len(self.payloads["model.safetensors"]))
            return original_response(request, **options)

        with patch.object(_download, "urlopen", side_effect=wrong_bytes), \
                self.assertRaisesRegex(RuntimeError, "SHA-256"):
            _download.release_files(self.root)
        self.assertFalse((self.cache / "model.safetensors").exists())
        self.assertEqual((self.cache / "config.json").read_bytes(), self.payloads["config.json"])
        self.assert_no_partial_files()

    def test_incomplete_and_oversized_responses_are_rejected(self):
        for data, message in ((b"{", "incomplete"), (b"x" * 100, "expected size")):
            with self.subTest(message=message), patch.object(_download, "urlopen", return_value=io.BytesIO(data)), \
                    self.assertRaisesRegex(RuntimeError, message):
                _download.release_files(self.root)
            self.assertFalse((self.cache / "config.json").exists())
            self.assert_no_partial_files()

    def test_an_interrupted_download_leaves_no_partial_file(self):
        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise URLError("connection interrupted")
                return super().read(3)

        with patch.object(_download, "urlopen", return_value=InterruptedResponse(b"partial")), \
                self.assertRaisesRegex(RuntimeError, "connection interrupted"):
            _download.release_files(self.root)
        self.assertFalse((self.cache / "config.json").exists())
        self.assert_no_partial_files()

    def test_http_404_is_a_readable_error_and_does_not_publish_a_file(self):
        error = HTTPError(_download.RELEASE_BASE_URL + "/config.json", 404, "Not Found", {}, None)
        with patch.object(_download, "urlopen", side_effect=error), \
                self.assertRaisesRegex(RuntimeError, "GitHub Release.*404"):
            _download.release_files(self.root)
        self.assertFalse((self.cache / "config.json").exists())
        self.assert_no_partial_files()

    def test_corrupt_cache_is_repaired_online(self):
        self.cached()
        weights = self.cache / "model.safetensors"
        weights.write_bytes(b"corrupt")
        with patch.object(_download, "urlopen", side_effect=self.response) as download:
            _download.release_files(self.root)
        self.assertEqual(download.call_count, 1)
        self.assertTrue(download.call_args.args[0].full_url.endswith("model.safetensors"))
        self.assertEqual(weights.read_bytes(), self.payloads["model.safetensors"])
        self.assert_no_partial_files()

    def test_failed_repair_preserves_the_completed_other_file(self):
        self.cached()
        config = self.cache / "config.json"
        identity = config.stat().st_ino
        weights = self.cache / "model.safetensors"
        weights.write_bytes(b"corrupt")
        with patch.object(_download, "urlopen", side_effect=URLError("connection interrupted")), \
                self.assertRaisesRegex(RuntimeError, "connection interrupted"):
            _download.release_files(self.root)
        self.assertEqual(config.read_bytes(), self.payloads["config.json"])
        self.assertEqual(config.stat().st_ino, identity)
        self.assertEqual(weights.read_bytes(), b"corrupt")
        self.assert_no_partial_files()

    def test_both_offline_modes_load_verified_files_without_network(self):
        self.cached()
        for flag, environment in ((True, "0"), (False, "1")):
            with self.subTest(local_files_only=flag, environment=environment), \
                    patch.dict(os.environ, {"HF_HUB_OFFLINE": environment}), \
                    patch.object(_download, "urlopen", side_effect=AssertionError("network forbidden")) as download:
                files = _download.release_files(self.root, local_files_only=flag)
                self.assertEqual(files, (str(self.cache / "config.json"), str(self.cache / "model.safetensors")))
                download.assert_not_called()

    def test_offline_cache_miss_or_corruption_never_contacts_network(self):
        for flag, environment in ((True, "0"), (False, "1")):
            with self.subTest(local_files_only=flag, environment=environment), \
                    patch.dict(os.environ, {"HF_HUB_OFFLINE": environment}), \
                    patch.object(_download, "urlopen", side_effect=AssertionError("network forbidden")) as download:
                with self.assertRaisesRegex(RuntimeError, "missing or corrupt.*offline"):
                    _download.release_files(self.root, local_files_only=flag)
                download.assert_not_called()
        self.assertFalse(self.cache.exists())
        self.cached()
        (self.cache / "model.safetensors").write_bytes(b"corrupt")
        with patch.object(_download, "urlopen", side_effect=AssertionError("network forbidden")) as download, \
                self.assertRaisesRegex(RuntimeError, "missing or corrupt"):
            _download.release_files(self.root, local_files_only=True)
        download.assert_not_called()

    def test_concurrent_downloads_publish_only_complete_verified_files(self):
        barrier = threading.Barrier(2)

        def response(request, **options):
            if request.full_url.endswith("config.json"):
                barrier.wait(timeout=10)
            return self.response(request, **options)

        with patch.object(_download, "urlopen", side_effect=response), ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(_download.release_files, self.root) for _ in range(2)]
            files = [task.result(timeout=20) for task in tasks]
        self.assertEqual(files[0], files[1])
        for name, data in self.payloads.items():
            self.assertEqual((self.cache / name).read_bytes(), data)
        self.assert_no_partial_files()


class ReleaseRoutingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.config = self.folder / "config.json"
        self.config.write_text(json.dumps({"model": {}, "preprocess": {}, "threshold": 0.5, "seed": 11}))
        self.weights = self.folder / "model.safetensors"
        self.weights.write_bytes(b"stand-in weights")
        self.files = str(self.config), str(self.weights)
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("safetensors.torch.load_file", return_value={}))
        self.patches.enter_context(patch.object(api, "PERSIST"))
        self.patches.enter_context(patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}))
        self.patches.enter_context(patch("huggingface_hub.constants.HF_HUB_OFFLINE", False))

    def test_default_uses_the_hub_first_and_does_not_touch_github_on_success(self):
        with patch.object(api, "_files", return_value=self.files) as hub, \
                patch.object(_download, "release_files", side_effect=AssertionError("fallback not needed")) as github:
            loaded = api.load(device="cpu", cache_dir="cache")
        hub.assert_called_once_with(api.DEFAULT_REPO, None, api.DEFAULT_REVISION, "cache", False)
        github.assert_not_called()
        self.assertEqual(loaded.source["model"], api.DEFAULT_REPO)
        self.assertNotIn("provider", loaded.source)

    def test_default_falls_back_and_records_the_actual_release_source(self):
        with patch.object(api, "_files", side_effect=RuntimeError("Hub unavailable")) as hub, \
                patch.object(_download, "release_files", return_value=self.files) as github:
            loaded = api.load(device="cpu", cache_dir="cache")
        hub.assert_called_once()
        github.assert_called_once_with("cache", local_files_only=False)
        self.assertEqual(loaded.source["provider"], "github-release")
        self.assertEqual(loaded.source["model"], "https://github.com/linty5/PERSIST/releases/tag/v1.1.0")
        self.assertEqual(loaded.source["revision"], _download.DEFAULT_RELEASE)
        self.assertEqual(loaded.source["sha256"], hashlib.sha256(self.weights.read_bytes()).hexdigest())

    def test_custom_repo_explicit_revision_variant_and_local_path_never_fall_back(self):
        cases = (("example/persist", None, None), (api.DEFAULT_REPO, "main", None),
                 (api.DEFAULT_REPO, "v1.1.0", None), (api.DEFAULT_REPO, None, "seed11"),
                 (str(self.folder), None, None))
        for model, revision, variant in cases:
            with self.subTest(model=model, revision=revision, variant=variant), \
                    patch.object(api, "_files", side_effect=RuntimeError("Hub unavailable")), \
                    patch.object(_download, "release_files") as github, \
                    self.assertRaisesRegex(RuntimeError, "Hub unavailable"):
                api.load(model, variant=variant, device="cpu", revision=revision)
            github.assert_not_called()

    def test_default_forwards_local_and_global_offline_modes_to_the_fallback(self):
        for local, global_offline in ((True, False), (False, True)):
            with self.subTest(local=local, global_offline=global_offline), \
                    patch("huggingface_hub.constants.HF_HUB_OFFLINE", global_offline), \
                    patch.object(api, "_files", side_effect=RuntimeError("not cached")), \
                    patch.object(_download, "release_files", return_value=self.files) as github:
                api.load(device="cpu", cache_dir="cache", local_files_only=local)
            github.assert_called_once_with("cache", local_files_only=True)

    def test_offline_environment_is_applied_to_both_sources_after_hub_import(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}), \
                patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("not cached")) as hub, \
                patch.object(_download, "urlopen", side_effect=AssertionError("network forbidden")) as github, \
                self.assertRaisesRegex(RuntimeError, "offline"):
            api.load(device="cpu", cache_dir=str(self.folder))
        self.assertTrue(all(call.kwargs["local_files_only"] for call in hub.call_args_list))
        github.assert_not_called()


if __name__ == "__main__":
    unittest.main()
