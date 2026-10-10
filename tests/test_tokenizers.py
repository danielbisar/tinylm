import io
import math
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from core import (
    BPETokenizer, ByteTokenizer, Config, ModelRepository, build_model,
    create_tokenizer, tokenizer_from_metadata,
)
from run import generate
import run
import train


class ConstantModel(torch.nn.Module):
    max_context = 8

    def __init__(self, vocab_size, token=None):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.token = token

    def forward(self, x):
        logits = torch.zeros(*x.shape, self.vocab_size)
        if self.token is not None:
            logits[..., self.token] = 100
        return logits, torch.tensor(0.0)


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tokenizer = BPETokenizer.train(
            ('Hello world! Hello there!\n hello hello\n' * 20), 300, 2,
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def small_config(self, **kwargs):
        return Config(
            dim=16, target_head_dim=8, context=8, batch_size=2,
            recurrent_steps=1, transform_layers=1, num_experts=2, top_k=1,
            **kwargs,
        )

    def test_byte_round_trip_and_all_byte_lengths(self):
        tokenizer = ByteTokenizer()
        text = 'Hällo 世界 🙂\n\x00'
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.encode(text), list(text.encode('utf-8')))
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(b''.join(tokenizer.token_bytes(i) for i in range(256)), bytes(range(256)))

    def test_bpe_ascii_round_trip_without_normalization(self):
        for text in ('Hello world!', '  Hello\tworld!\n\n ', '', '\r\n'):
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_bpe_unicode_and_unseen_text_have_no_unknown_tokens(self):
        for text in ('Hällo 世界 🙂', '\x00\x01\x7f 🦄 e\u0301\r\n', ''.join(chr(i) for i in range(256))):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertEqual(self.tokenizer.decode(ids), text)
                self.assertEqual(b''.join(self.tokenizer.token_bytes(i) for i in ids), text.encode('utf-8'))
        self.assertIsNone(self.tokenizer.tokenizer.token_to_id('<unk>'))
        self.assertEqual(len(self.tokenizer.byte_lengths), self.tokenizer.vocab_size)

    def test_tokenizer_factory_and_invalid_settings(self):
        self.assertIsInstance(create_tokenizer(Config(), ''), ByteTokenizer)
        self.assertIsInstance(create_tokenizer(Config(tokenizer_type='bpe'), 'hello ' * 10), BPETokenizer)
        with self.assertRaises(ValueError):
            create_tokenizer(Config(tokenizer_type='unknown'), '')
        with self.assertRaises(ValueError):
            BPETokenizer.train('hello', 255)

    def test_nonbyte_vocabulary_forward_backward(self):
        config = self.small_config(vocab_size=300)
        model = build_model(config.model_config(), 'cpu')
        tokens = torch.tensor([[256, 299, 0]])
        logits, aux = model(tokens)
        self.assertEqual(logits.shape, (1, 3, 300))
        (train.language_model_loss(logits, tokens) + .01 * aux).backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertEqual(model.output.out_features, 300)

    def test_bpe_checkpoint_is_self_contained_and_restores_identical_ids(self):
        tokenizer = self.tokenizer
        config = self.small_config(tokenizer_type='bpe', vocab_size=tokenizer.vocab_size)
        model = build_model(config.model_config(), 'cpu')
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            repository = ModelRepository(directory)
            repository.save(repository.best_filename, model, optimizer, 1, config, 2.0, tokenizer, 1.0)
            restored_model, checkpoint = repository.load(repository.best_filename, 'cpu')
        restored = tokenizer_from_metadata(checkpoint['tokenizer'])
        text = 'Hällo 世界 🙂\nHello there!'
        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(restored.byte_lengths, tokenizer.byte_lengths)
        self.assertEqual(checkpoint['selection_metric'], 'bpb')
        ids = torch.tensor([tokenizer.encode('Hello')])
        torch.testing.assert_close(restored_model(ids)[0], model(ids)[0])

    def test_legacy_byte_checkpoint_without_tokenizer_or_vocab_size(self):
        config = self.small_config()
        model = build_model(config.model_config(), 'cpu')
        legacy_config = config.model_config()
        legacy_config.pop('vocab_size')
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            repository = ModelRepository(directory)
            torch.save({'model': model.state_dict(), 'config': legacy_config, 'step': 1}, Path(directory) / repository.best_filename)
            restored_model, checkpoint = repository.load(repository.best_filename, 'cpu')
        self.assertEqual(restored_model.output.out_features, 256)
        self.assertEqual(checkpoint['config']['vocab_size'], 256)

    def test_checkpoint_rejects_mismatched_vocabulary(self):
        config = self.small_config(vocab_size=300)
        model = build_model(config.model_config(), 'cpu')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ModelRepository(directory).save('bad.pt', model, torch.optim.AdamW(model.parameters()), 1, config, 1.0)

    def test_byte_bpb_equals_token_cross_entropy_in_bits(self):
        config = self.small_config()
        data = torch.tensor(ByteTokenizer().encode('a small known example'))
        model = ConstantModel(256)
        loss, bpb = train.validation_loss(model, data, config, 'cpu', 2)
        self.assertAlmostEqual(loss, math.log(256), places=5)
        self.assertAlmostEqual(bpb, loss / math.log(2), places=6)
        self.assertTrue(model.training)
        self.assertAlmostEqual(train.bits_per_byte(3 * math.log(2), 3), 1.0)

    def test_bpe_exact_byte_accounting_for_shifted_windows(self):
        tokenizer = self.tokenizer
        text = 'Hello world!\n Hällo 世界 🙂\nhello\n\nhello'
        ids = tokenizer.encode(text)
        lengths = torch.tensor(tokenizer.byte_lengths)
        self.assertEqual(int(lengths[ids].sum()), len(text.encode('utf-8')))
        # Includes Unicode byte fragments and newline boundaries. Force two
        # differently sized target windows, so the global average is wrong.
        context = 3
        starts = [0, len(ids) - context - 1]
        config = self.small_config(vocab_size=tokenizer.vocab_size)
        config.context = context
        config.batch_size = 1
        data = torch.tensor(ids)
        target_bytes = sum(int(lengths[ids[s + 1:s + context + 1]].sum()) for s in starts)
        with patch('train.torch.randint', side_effect=[torch.tensor([s]) for s in starts]):
            loss, bpb = train.validation_loss(ConstantModel(tokenizer.vocab_size), data, config, 'cpu', 2, tokenizer.byte_lengths)
        self.assertAlmostEqual(bpb, loss * 2 * context / (math.log(2) * target_bytes), places=6)

    def test_generation_stops_by_bytes_for_multibyte_bpe_token(self):
        tokenizer = self.tokenizer
        token = max(range(tokenizer.vocab_size), key=lambda i: tokenizer.byte_lengths[i])
        payload = tokenizer.token_bytes(token)
        self.assertGreater(len(payload), 1)
        model = ConstantModel(tokenizer.vocab_size, token)
        text, tokens, byte_count, elapsed = generate(model, tokenizer, 'Hello', len(payload) + 1)
        self.assertEqual(tokens, 2)
        self.assertEqual(byte_count, len(payload) * 2)
        self.assertEqual(text, 'Hello' + (payload * 2).decode('utf-8'))
        self.assertGreater(elapsed, 0)

    def test_generation_incrementally_decodes_split_unicode(self):
        class SequenceModel(ConstantModel):
            def __init__(self):
                super().__init__(256)
                self.steps = iter('🙂'.encode('utf-8'))

            def forward(self, x):
                self.token = next(self.steps)
                return super().forward(x)

        text, tokens, byte_count, _ = generate(SequenceModel(), ByteTokenizer(), 'é', 4)
        self.assertEqual((text, tokens, byte_count), ('é🙂', 4, 4))

    def test_bpe_workflow_trains_only_on_training_source(self):
        training_text = 'Hello hello world!\n' * 40
        validation_text = 'Unseen 世界 🙂 validation!\n' * 20
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = self.small_config(tokenizer_type='bpe', bpe_vocab_size=300, steps=1, validate_every=1, validation_batches=1, final_validation_batches=1)
            repository = ModelRepository(directory)
            with patch('train.load_source', side_effect=[training_text, validation_text]), patch('train.create_tokenizer', wraps=create_tokenizer) as factory, patch.object(torch.cuda, 'is_available', return_value=False):
                train.train(config, repository, time.perf_counter())
            factory.assert_called_once_with(config, training_text)
            model, checkpoint = repository.load(repository.last_filename, 'cpu')
            tokenizer = tokenizer_from_metadata(checkpoint['tokenizer'])
            self.assertEqual(checkpoint['data_metadata']['validation']['raw_utf8_bytes'], len(validation_text.encode('utf-8')))
            self.assertEqual(tokenizer.decode(tokenizer.encode(validation_text)), validation_text)
            result = generate(model, tokenizer, 'Hello', 5)
            self.assertGreaterEqual(result[2], 5)
            # Editable defaults deliberately say byte/256; generation must
            # infer BPE and its model vocabulary entirely from the checkpoint.
            generation_config = Config(max_output_bytes=5, prompts=("Hällo 世界 🙂",))
            output = io.StringIO()
            with patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(output):
                run.run(generation_config, repository, repository.last_filename)
            self.assertIn("Tokenizer: bpe", output.getvalue())
            self.assertIn("tokens/sec", output.getvalue())
            self.assertIn("UTF-8 bytes", output.getvalue())


if __name__ == '__main__':
    unittest.main()
