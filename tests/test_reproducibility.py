import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from core import (
    ByteTokenizer, Config, build_model, log_run, print_metadata,
    run_metadata, source_provenance,
)
import train


class ReproducibilityTests(unittest.TestCase):
    def config(self):
        return Config(
            dim=16, target_head_dim=8, context=4, batch_size=2,
            transform_layers=1, recurrent_steps=1, num_experts=2, top_k=1,
            vocab_size=257,
        )

    def test_batch_includes_last_valid_window(self):
        data = torch.arange(7)
        with patch("train.torch.randint", return_value=torch.tensor([2])) as sample:
            x, y = train.get_batch(data, 1, 4, "cpu")
        sample.assert_called_once_with(0, 3, (1,), generator=None)
        self.assertEqual(x.tolist(), [[2, 3, 4, 5]])
        self.assertEqual(y.tolist(), [[3, 4, 5, 6]])

    def test_exactly_context_plus_one_tokens_is_a_valid_batch(self):
        x, y = train.get_batch(torch.arange(5), 2, 4, "cpu")
        self.assertEqual(x.tolist(), [[0, 1, 2, 3]] * 2)
        self.assertEqual(y.tolist(), [[1, 2, 3, 4]] * 2)
        for size in (0, 4):
            with self.subTest(size=size), self.assertRaises(ValueError):
                train.get_batch(torch.arange(size), 1, 4, "cpu")

    def test_invalid_and_zero_step_budgets_raise_explicitly(self):
        for budget in (0, -1, 1, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                train.configure_training_budget(Config(target_training_bytes=budget), 1)
        for average in (0, -1, float("nan"), float("inf")):
            with self.subTest(average=average), self.assertRaises(ValueError):
                train.configure_training_budget(Config(), average)
        with self.assertRaisesRegex(ValueError, "at least one training step"):
            train.configure_training_budget(Config(target_training_bytes=None, steps=0), 1)

    def test_eos_changes_effective_byte_budget_steps(self):
        tokenizer = ByteTokenizer()
        data = torch.tensor(train.encode_documents(["a", "b"], tokenizer))
        average = train.source_statistics(data, tokenizer)["average_bytes_per_token"]
        config = Config()
        train.configure_training_budget(config, average)
        self.assertEqual(average, 0.5)
        self.assertEqual(config.steps, 20000)

    def test_validation_has_fixed_prefix_and_preserves_rng_streams(self):
        config = self.config()
        model = build_model(config.model_config(), "cpu")
        data = torch.tensor(train.encode_documents(["abcdefg"] * 10, ByteTokenizer()))
        generator = torch.Generator().manual_seed(config.seed)
        training_state = generator.get_state().clone()
        global_state = torch.get_rng_state().clone()
        windows = []
        get_batch = train.get_batch

        def capture(*args, **kwargs):
            result = get_batch(*args, **kwargs)
            windows.append(result[0].clone())
            return result

        with patch.object(train, "get_batch", side_effect=capture):
            first = train.validation_loss(
                model, data, config, "cpu", 2, ByteTokenizer().byte_lengths,
                eos_token_id=256, return_metrics=True,
            )
            repeated = train.validation_loss(
                model, data, config, "cpu", 2, ByteTokenizer().byte_lengths,
                eos_token_id=256, return_metrics=True,
            )
            train.validation_loss(
                model, data, config, "cpu", 4, ByteTokenizer().byte_lengths,
                eos_token_id=256,
            )
        self.assertEqual(first, repeated)
        for i in range(2):
            torch.testing.assert_close(windows[i], windows[i + 2])
            torch.testing.assert_close(windows[i], windows[i + 4])
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(training_state, generator.get_state()))
        self.assertTrue(model.training)

    def test_training_stream_is_independent_of_global_rng(self):
        data = torch.arange(100)
        generators = [torch.Generator().manual_seed(42) for _ in range(2)]
        for _ in range(3):
            first = train.get_batch(data, 2, 4, "cpu", generator=generators[0])
            torch.rand(100)
            second = train.get_batch(data, 2, 4, "cpu", generator=generators[1])
            torch.testing.assert_close(first[0], second[0])
            torch.testing.assert_close(first[1], second[1])

    def test_dataset_version_fingerprint_and_requested_revision_are_captured(self):
        dataset = SimpleNamespace(
            info=SimpleNamespace(version="1.2.3"), _fingerprint="encoded-split-id",
        )

        class Dataset:
            info = dataset.info
            _fingerprint = dataset._fingerprint

            def __getitem__(self, key):
                return ["document"]

        metadata = {}
        config = Config(dataset_revision="immutable-revision")
        with patch("datasets.load_dataset", return_value=Dataset()) as load:
            with redirect_stdout(io.StringIO()):
                documents = train.load_source(config, config.train_split, metadata)
        self.assertEqual(documents, ["document"])
        load.assert_called_once_with(
            config.dataset_name, split=config.train_split, revision="immutable-revision",
        )
        self.assertEqual(metadata[config.train_split], {
            "version": "1.2.3", "fingerprint": "encoded-split-id",
        })

    def test_git_provenance_has_exact_hashes_and_unknown_state_is_explicit(self):
        results = [
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n"),
            subprocess.CompletedProcess([], 0, stdout=" M core.py\n"),
        ]
        with patch("core.subprocess.run", side_effect=results):
            metadata = source_provenance()
        self.assertEqual(metadata["git_commit"], "a" * 40)
        self.assertTrue(metadata["git_dirty"])
        self.assertTrue(all(len(value) == 64 for value in metadata["source_sha256"].values()))
        output = io.StringIO()
        with patch("core.subprocess.run", side_effect=FileNotFoundError), redirect_stdout(output):
            metadata = source_provenance()
        self.assertIsNone(metadata["git_commit"])
        self.assertIsNone(metadata["git_dirty"])
        self.assertIn("WARNING", output.getvalue())

    def test_public_log_redacts_paths_and_does_not_capture_argv_secrets(self):
        root = Path(__file__).resolve().parents[1]
        config = self.config()
        config.model_directory = str(root / "model")
        metadata = run_metadata(config, "cpu", tokenizer=ByteTokenizer())
        self.assertEqual(metadata["config"]["model_directory"], "<repo>/model")
        self.assertEqual(metadata["tokenizer"]["eos_token_id"], 256)
        self.assertEqual(metadata["cuda_runtime"], torch.version.cuda)
        self.assertEqual(metadata["libraries"]["torch"], str(torch.__version__))
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(sys, "argv", ["train.py", "--access-token=do-not-record"]),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with log_run(tmp, "test-", "test", provenance={"git_dirty": False}) as path:
                        print(root / "core.py")
                        print(Path.home() / "private-location")
                        print_metadata("Metadata", metadata)
                        raise RuntimeError("test failure")
            text = path.read_text()
        self.assertNotIn(str(root), text)
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn("do-not-record", text)
        self.assertIn("<repo>/core.py", text)
        self.assertIn("<home>/private-location", text)
        self.assertIn("RuntimeError: test failure", text)
        self.assertIn(json.dumps(False), text)


if __name__ == "__main__":
    unittest.main()
