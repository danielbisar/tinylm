import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from core import ByteTokenizer, Config, ModelRepository, TrainingTimer, log_run
import run
import train


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, directory):
        return Config(
            dim=16,
            target_head_dim=8,
            context=8,
            batch_size=2,
            recurrent_steps=1,
            transform_layers=1,
            num_experts=2,
            top_k=1,
            steps=2,
            target_training_bytes=None,
            validate_every=1,
            validation_batches=1,
            final_validation_batches=3,
            max_output_bytes=3,
            prompts=("é",),
            model_directory=str(directory / "model"),
            log_directory=str(directory / "docs/runs"),
        )

    def test_either_checkpoint_aborts_before_data_loading(self):
        cases = [("best",), ("last",), ("best", "last")]
        for names in cases:
            with (
                self.subTest(names=names),
                tempfile.TemporaryDirectory() as tmp,
            ):
                config = self.config(Path(tmp))
                model_directory = Path(config.model_directory)
                model_directory.mkdir()
                for name in names:
                    path = model_directory / f"test-{name}_model.pt"
                    path.write_bytes(b"old")
                output = io.StringIO()
                with patch.object(train, "Config", return_value=config):
                    with patch.object(train, "load_data") as load_data:
                        with redirect_stdout(output):
                            result = train.main(["--model-prefix=test-"])
                self.assertEqual(result, 1)
                load_data.assert_not_called()
                self.assertIn("WARNING", output.getvalue())
                for name in names:
                    path = model_directory / f"test-{name}_model.pt"
                    self.assertEqual(path.read_bytes(), b"old")
                logs = list(Path(config.log_directory).glob("*.md"))
                self.assertEqual(len(logs), 1)
                self.assertIn("Training aborted", logs[0].read_text())

    def test_train_force_overwrite_and_generate_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            model_directory = Path(config.model_directory)
            model_directory.mkdir()
            for name in ("best", "last"):
                (model_directory / f"test-{name}_model.pt").write_bytes(b"old")
            data = torch.arange(256, dtype=torch.long).repeat(2)
            output = io.StringIO()
            with patch.object(torch.cuda, "is_available", return_value=False):
                with patch.object(train, "Config", return_value=config):
                    with patch.object(train, "load_source", return_value="synthetic training text"):
                        with patch.object(train, "load_data", return_value=data):
                            with patch.object(
                                train,
                                "validation_loss",
                                wraps=train.validation_loss,
                            ) as validate:
                                with redirect_stdout(output):
                                    result = train.main([
                                        "--model-prefix=test-", "--force-overwrite",
                                    ])
                self.assertEqual(result, 0)
                self.assertEqual(
                    [call.args[4] for call in validate.call_args_list],
                    [1, 1, 3],
                )
                last = torch.load(model_directory / "test-last_model.pt")
                self.assertEqual(last["step"], 2)
                self.assertEqual(last["run_config"]["steps"], 2)
                self.assertEqual(last["run_config"]["validate_every"], 1)
                self.assertIsNone(last["run_config"]["target_training_bytes"])
                self.assertEqual(last["config"]["dim"], 16)
                self.assertEqual(
                    last["run_config"]["final_validation_batches"], 3,
                )
                self.assertEqual(last["tokenizer"], ByteTokenizer().metadata)
                self.assertEqual(last["config"]["vocab_size"], 257)
                self.assertIn("optimizer", last)

                # Generation must use checkpoint dimensions, even after the
                # editable default configuration changes between invocations.
                config.dim = 32
                for name in ("best", "last"):
                    with patch.object(run, "Config", return_value=config):
                        with redirect_stdout(output):
                            result = run.main([
                                "--model-prefix=test-", f"--checkpoint={name}",
                            ])
                    self.assertEqual(result, 0)

            text = output.getvalue()
            self.assertIn("Training time (steps only)", text)
            self.assertIn("Total runtime", text)
            self.assertIn("estimated remaining training loop", text)
            self.assertIn("output bytes/sec", text)
            self.assertIn("Dim: 16; heads: 2", text)
            logs = sorted(Path(config.log_directory).glob("*.md"))
            self.assertEqual(len(logs), 3)
            self.assertTrue(
                all(path.name.endswith("-test-.md") for path in logs)
            )
            self.assertIn("Results", logs[0].read_text())
            self.assertIn("Generation total", logs[-1].read_text())

    def test_training_time_excludes_gaps_between_intervals(self):
        timer = TrainingTimer("cpu")
        with patch("core.time.perf_counter", side_effect=[10, 15, 100, 102]):
            with timer.measure():
                pass
            # Time jumps forward for evaluation/checkpoint work before the
            # next timed training interval; that gap must not be counted.
            with timer.measure():
                pass
        self.assertEqual(timer.elapsed, 7)

    def test_byte_budget_matches_10000_step_baseline(self):
        config = Config(steps=17, validate_every=3)
        data = torch.tensor(ByteTokenizer().encode("Hällo 世界 🙂\n"))
        average = train.source_statistics(data, ByteTokenizer())["average_bytes_per_token"]
        train.configure_training_budget(config, average)
        self.assertEqual(config.steps, 10000)
        self.assertEqual(config.validate_every, 1000)
        self.assertEqual(config.steps * config.batch_size * config.context, 81_920_000)
        output = io.StringIO()
        with redirect_stdout(output):
            train.print_training_config(config, average)
        for line in (
            "Target source-byte budget: 81,920,000",
            "Effective training steps: 10,000",
            "Tokens per step: 8,192",
            "Estimated source bytes processed: 81,920,000.00",
            "Validate every: 1,000 steps",
        ):
            self.assertIn(line, output.getvalue())

    def test_validation_interval_follows_short_effective_run(self):
        config = Config(target_training_bytes=7 * 8192, validate_every=1000)
        train.configure_training_budget(config, 1.0)
        self.assertEqual(config.steps, 7)
        self.assertEqual(config.validate_every, 1)

    def test_disabled_budget_preserves_manual_settings(self):
        config = Config(target_training_bytes=None, steps=123, validate_every=7)
        train.configure_training_budget(config, 3.7)
        self.assertEqual(config.steps, 123)
        self.assertEqual(config.validate_every, 7)

    def test_generation_counts_displayed_utf8_bytes_excluding_prompt(self):
        class InvalidByteModel(torch.nn.Module):
            max_context = 8

            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                logits = torch.zeros(*x.shape, 256)
                logits[..., 255] = 1
                return logits, torch.tensor(0.0)

        text, generated_tokens, generated_bytes, elapsed = run.generate(
            InvalidByteModel(), ByteTokenizer(), "é", 3, greedy=True,
        )
        self.assertEqual(text, "é" + "\ufffd")
        self.assertEqual(generated_tokens, 1)
        self.assertEqual(generated_bytes, 3)
        self.assertGreater(len(text.encode("utf-8")), generated_bytes)
        self.assertGreater(elapsed, 0)

    def test_log_flushes_stdout_and_stderr_before_run_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(output), redirect_stderr(errors):
                with log_run(tmp, "test-", "test") as path:
                    print("stdout message")
                    print("stderr message", file=sys.stderr)
                    self.assertIn("stdout message", path.read_text())
                    self.assertIn("stderr message", path.read_text())
            self.assertEqual(output.getvalue().count("stdout message"), 1)
            self.assertEqual(errors.getvalue().count("stderr message"), 1)
            self.assertTrue(path.read_text().endswith("\n```\n"))

    def test_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            repository = ModelRepository(config.model_directory, "test-")
            repository.directory.mkdir()
            path = repository.directory / repository.best_filename
            path.write_bytes(b"previous checkpoint")
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            with patch("core.torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    repository.save(
                        repository.best_filename,
                        model,
                        optimizer,
                        1,
                        config,
                        0.5,
                    )
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(repository.directory.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
