import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from core import ByteTokenizer, Config
import run


class SamplingModel(torch.nn.Module):
    max_context = 8

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = torch.full((*x.shape, 256), -float("inf"), device=x.device)
        logits[..., 97:100] = torch.tensor([0.6, 0.25, 0.15], device=x.device).log()
        return logits, torch.tensor(0.0)


class SamplingTests(unittest.TestCase):
    def generate(self, **kwargs):
        return run.generate(SamplingModel(), ByteTokenizer(), "prompt", 32, **kwargs)

    def test_default_and_explicit_greedy_do_not_sample(self):
        with patch.object(torch, "multinomial") as sample:
            self.assertEqual(self.generate()[0], "prompt" + "a" * 32)
            self.assertEqual(
                self.generate(greedy=True, temperature=2, top_p=1)[0],
                "prompt" + "a" * 32,
            )
        sample.assert_not_called()

    def test_sampling_is_reproducible_and_varies_with_seed(self):
        results = [
            self.generate(
                greedy=False, temperature=1, top_p=1,
                generator=torch.Generator().manual_seed(seed),
            )[:3]
            for seed in (42, 42, 43)
        ]
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0][0], results[2][0])
        self.assertEqual(results[0][1:], (32, 32))
        self.assertGreater(len(set(results[0][0][6:])), 1)

    def test_nucleus_keeps_threshold_crossing_token(self):
        with patch.object(torch, "multinomial", wraps=torch.multinomial) as sample:
            text = self.generate(greedy=False, temperature=1, top_p=0.7)[0]
        probabilities = sample.call_args.args[0]
        torch.testing.assert_close(
            probabilities[0, :3], torch.tensor([0.6 / 0.85, 0.25 / 0.85, 0]),
        )
        self.assertEqual(torch.count_nonzero(probabilities).item(), 2)
        self.assertTrue(set(text[6:]) <= {"a", "b"})
        self.assertEqual(self.generate(greedy=False, top_p=0.01)[0], "prompt" + "a" * 32)

    def test_temperature_scales_full_vocabulary_probabilities(self):
        with patch.object(torch, "multinomial", wraps=torch.multinomial) as sample:
            self.generate(greedy=False, temperature=0.5, top_p=1)
        expected = torch.tensor([0.6, 0.25, 0.15]).square()
        expected /= expected.sum()
        torch.testing.assert_close(sample.call_args.args[0][0, 97:100], expected)

    def test_run_seeds_sampling_and_reports_settings(self):
        config = Config(seed=7, prompts=("prompt",), max_output_bytes=32)
        repository = Mock()
        repository.load.return_value = (SamplingModel(), {"config": {}, "step": 1})
        texts = []
        generate = run.generate

        def capture(*args, **kwargs):
            result = generate(*args, **kwargs)
            texts.append(result[0])
            return result

        output = io.StringIO()
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(run, "print_model_info"),
            patch.object(run, "generate", side_effect=capture),
            redirect_stdout(output),
        ):
            for _ in range(2):
                run.run(config, repository, "best_model.pt", greedy=False)
        self.assertEqual(texts[0], texts[1])
        self.assertIn("Sampling; temperature=0.8; top-p=0.9; seed=7", output.getvalue())

    def test_cli_modes_and_seed(self):
        cases = (
            ([], True, 0.8, 0.9, 42),
            (["--seed=7"], True, 0.8, 0.9, 7),
            (["--temperature=0.8", "--top-p=0.9", "--seed=42"], False, 0.8, 0.9, 42),
            (["--temperature=1.2"], False, 1.2, 0.9, 42),
            (["--top-p=1"], False, 0.8, 1, 42),
            (["--greedy", "--temperature=0.8", "--top-p=0.9"], True, 0.8, 0.9, 42),
        )
        for arguments, greedy, temperature, top_p, seed in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as tmp:
                config = Config(model_directory=tmp, log_directory=str(Path(tmp) / "logs"))
                with (
                    patch.object(run, "Config", return_value=config),
                    patch.object(run, "run") as generate_run,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(run.main(arguments), 0)
                self.assertEqual(config.seed, seed)
                self.assertEqual(generate_run.call_args.kwargs, {
                    "greedy": greedy, "temperature": temperature, "top_p": top_p,
                })

    def test_invalid_sampling_arguments_fail_before_loading_checkpoint(self):
        for flag, values in (
            ("--temperature", ("0", "-1", "nan", "inf")),
            ("--top-p", ("0", "-1", "1.1", "nan", "inf")),
        ):
            for value in values:
                with self.subTest(flag=flag, value=value), patch.object(run, "ModelRepository") as repository:
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        run.main([f"{flag}={value}"])
                    self.assertEqual(error.exception.code, 2)
                    repository.assert_not_called()
        for kwargs in ({"temperature": 0}, {"top_p": float("nan")}):
            with self.assertRaises(ValueError):
                self.generate(greedy=False, **kwargs)


if __name__ == "__main__":
    unittest.main()
