import io
import math
import random
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
        self.assertEqual(tokenizer.vocab_size, 257)
        self.assertEqual(b''.join(tokenizer.token_bytes(i) for i in range(256)), bytes(range(256)))

    def test_eos_is_special_and_has_zero_byte_payload(self):
        from tokenizers import Tokenizer

        tokenizer = self.tokenizer
        eos = tokenizer.eos_token_id
        self.assertEqual(tokenizer.eos_token, '<|endoftext|>')
        self.assertEqual(tokenizer.tokenizer.token_to_id(tokenizer.eos_token), eos)
        hf_tokenizer = Tokenizer.from_str(tokenizer.metadata['serialization'])
        hf_tokenizer.encode_special_tokens = False
        self.assertEqual(hf_tokenizer.encode(tokenizer.eos_token).ids, [eos])
        self.assertTrue(hf_tokenizer.get_added_tokens_decoder()[eos].special)
        for tokenizer in (tokenizer, ByteTokenizer()):
            self.assertEqual(tokenizer.token_bytes(tokenizer.eos_token_id), b'')
            self.assertEqual(tokenizer.byte_lengths[tokenizer.eos_token_id], 0)
            self.assertEqual(tokenizer.decode([tokenizer.eos_token_id]), '')
            literal = tokenizer.eos_token
            self.assertNotIn(tokenizer.eos_token_id, tokenizer.encode(literal))
            self.assertEqual(tokenizer.decode(tokenizer.encode(literal)), literal)

    def test_bpe_8192_is_total_vocabulary_including_eos(self):
        rng = random.Random(42)
        documents = [''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=20))
                     for _ in range(10000)]
        tokenizer = BPETokenizer.train(documents, 8192, 1)
        self.assertEqual(tokenizer.vocab_size, 8192)
        config = self.small_config(vocab_size=tokenizer.vocab_size)
        model = build_model(config.model_config(), 'cpu')
        self.assertEqual(model.token_embedding.num_embeddings, 8192)
        self.assertEqual(model.output.out_features, 8192)
        from tokenizers.pre_tokenizers import ByteLevel
        raw_tokens = [tokenizer.tokenizer.token_to_id(symbol)
                      for symbol in ByteLevel.alphabet()]
        self.assertEqual(set(tokenizer.token_bytes(i) for i in raw_tokens),
                         {bytes([i]) for i in range(256)})
        with self.assertRaises(ValueError):
            BPETokenizer.train(documents, 256)

    def test_documents_have_one_eos_and_exact_source_statistics(self):
        documents = ['Hello', '', '世界', '<|endoftext|>', '\x00']
        for tokenizer in (self.tokenizer, ByteTokenizer()):
            with self.subTest(type=tokenizer.metadata['type']):
                ids = train.encode_documents(iter(documents), tokenizer)
                expected = [token for doc in documents
                            for token in tokenizer.encode(doc) + [tokenizer.eos_token_id]]
                self.assertEqual(ids, expected)
                self.assertEqual(ids.count(tokenizer.eos_token_id), len(documents))
                raw_bytes = sum(len(doc.encode('utf-8')) for doc in documents)
                statistics = train.source_statistics(torch.tensor(ids), tokenizer)
                self.assertEqual(statistics['raw_utf8_bytes'], raw_bytes)
                self.assertEqual(statistics['average_bytes_per_token'], raw_bytes / len(ids))
                with redirect_stdout(io.StringIO()):
                    data = train.load_data(Config(), 'unused', tokenizer, documents)
                self.assertEqual(data.tolist(), ids)
                self.assertEqual(train.source_statistics(torch.tensor([], dtype=torch.long), tokenizer)['average_bytes_per_token'], 0)

    def test_bpe_training_and_encoding_do_not_merge_across_documents(self):
        tokenizer = BPETokenizer.train(['a', 'b'] * 20, 300, 1)
        self.assertNotIn('ab', tokenizer.tokenizer.get_vocab())
        ids = train.encode_documents(['a', 'b'], tokenizer)
        self.assertEqual(ids, tokenizer.encode('a') + [tokenizer.eos_token_id]
                         + tokenizer.encode('b') + [tokenizer.eos_token_id])

    def test_load_source_preserves_dataset_items(self):
        documents = ['first', '', 'second\nline']
        with patch('datasets.load_dataset', return_value={'text': documents}) as load:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(train.load_source(Config(dataset_name='generic'), 'train'), documents)
        load.assert_called_once_with('generic', split='train')

    def test_eos_byte_metadata_roundtrip_and_legacy_metadata(self):
        tokenizer = ByteTokenizer()
        restored = tokenizer_from_metadata(tokenizer.metadata)
        self.assertEqual(restored.eos_token_id, 256)
        self.assertEqual(restored.byte_lengths, tokenizer.byte_lengths)
        self.assertEqual(restored.metadata, tokenizer.metadata)
        legacy = tokenizer_from_metadata({'type': 'byte', 'encoding': 'utf-8', 'vocab_size': 256})
        self.assertIsNone(legacy.eos_token_id)
        self.assertEqual(legacy.vocab_size, 256)
        for i in range(256):
            self.assertEqual(tokenizer.token_bytes(i), bytes([i]))
            self.assertEqual(legacy.token_bytes(i), bytes([i]))

    def test_legacy_bpe_checkpoint_without_eos(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        hf_tokenizer = Tokenizer(models.BPE())
        hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tokenizer.decoder = decoders.ByteLevel()
        hf_tokenizer.train_from_iterator(['hello world ' * 20], trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ))
        metadata = {'type': 'bpe', 'vocab_size': hf_tokenizer.get_vocab_size(),
                    'training_config': {'vocab_size': 300, 'min_frequency': 2},
                    'serialization': hf_tokenizer.to_str()}
        tokenizer = tokenizer_from_metadata(metadata)
        self.assertIsNone(tokenizer.eos_token_id)
        config = self.small_config(vocab_size=tokenizer.vocab_size)
        model = build_model(config.model_config(), 'cpu')
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            repository = ModelRepository(directory)
            torch.save({'model': model.state_dict(), 'config': config.model_config(),
                        'step': 1, 'tokenizer': metadata},
                       Path(directory) / repository.best_filename)
            restored, checkpoint = repository.load(repository.best_filename, 'cpu')
        tokenizer = tokenizer_from_metadata(checkpoint['tokenizer'])
        token = tokenizer.encode('x')[0]
        result = generate(ConstantModel(tokenizer.vocab_size, token), tokenizer,
                          'hello', 3, greedy=True, return_stop_reason=True)
        self.assertEqual(result[0], 'helloxxx')
        self.assertEqual(result[-1], 'max_output_bytes')
        self.assertEqual(restored.output.out_features, tokenizer.vocab_size)

    def test_eos_bpb_accounts_for_target_bytes_only(self):
        tokenizer = ByteTokenizer()
        config = self.small_config(vocab_size=257)
        config.context = 2
        config.batch_size = 1
        data = torch.tensor(train.encode_documents(['ab', 'c'], tokenizer))
        # Targets are b, EOS: two predictions but just one source byte.
        with patch('train.torch.randint', return_value=torch.tensor([0])):
            loss, bpb = train.validation_loss(ConstantModel(257), data, config, 'cpu', 1, tokenizer.byte_lengths)
        self.assertAlmostEqual(bpb, 2 * loss / math.log(2))
        eos_only = torch.full((10,), tokenizer.eos_token_id)
        loss, bpb = train.validation_loss(ConstantModel(257), eos_only, config, 'cpu', 1, tokenizer.byte_lengths)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(bpb, float('inf'))
        with self.assertRaises(ValueError):
            train.configure_training_budget(Config(), 0)

    def test_generation_stops_at_eos_before_payload_lookup(self):
        for tokenizer in (ByteTokenizer(), self.tokenizer):
            for greedy in (True, False):
                with self.subTest(type=tokenizer.metadata['type'], greedy=greedy):
                    model = ConstantModel(tokenizer.vocab_size, tokenizer.eos_token_id)
                    with patch.object(tokenizer, 'token_bytes', side_effect=AssertionError('EOS payload lookup')):
                        result = generate(model, tokenizer, 'prompt', 10, greedy=greedy, return_stop_reason=True)
                    self.assertEqual(result[:3], ('prompt', 0, 0))
                    self.assertEqual(result[-1], 'EOS')

    def test_generation_emits_text_then_eos_or_byte_limit(self):
        tokenizer = ByteTokenizer()

        class SequenceModel(ConstantModel):
            def __init__(self):
                super().__init__(tokenizer.vocab_size)
                self.steps = iter([ord('a'), tokenizer.eos_token_id])

            def forward(self, x):
                self.token = next(self.steps)
                return super().forward(x)

        result = generate(SequenceModel(), tokenizer, 'p', 10, greedy=True, return_stop_reason=True)
        self.assertEqual(result[:3], ('pa', 1, 1))
        self.assertEqual(result[-1], 'EOS')
        result = generate(SequenceModel(), tokenizer, 'p', 1, greedy=True, return_stop_reason=True)
        self.assertEqual(result[:3], ('pa', 1, 1))
        self.assertEqual(result[-1], 'max_output_bytes')

    def test_eos_flushes_preceding_incomplete_utf8_without_decoding_eos(self):
        tokenizer = ByteTokenizer()

        class SequenceModel(ConstantModel):
            def __init__(self):
                super().__init__(tokenizer.vocab_size)
                self.steps = iter([0xf0, tokenizer.eos_token_id])

            def forward(self, x):
                self.token = next(self.steps)
                return super().forward(x)

        result = generate(SequenceModel(), tokenizer, 'p', 10, greedy=True, return_stop_reason=True)
        self.assertEqual(result[:3], ('p\ufffd', 1, 3))
        self.assertEqual(result[-1], 'EOS')

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
        self.assertEqual(restored.eos_token_id, tokenizer.eos_token_id)
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
        legacy = tokenizer_from_metadata({'type': 'byte', 'vocab_size': 256})
        result = generate(ConstantModel(256, 0), legacy, 'p', 2, greedy=True, return_stop_reason=True)
        self.assertEqual(result[:3], ('p\x00\x00', 2, 2))
        self.assertEqual(result[-1], 'max_output_bytes')

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

    def test_bpe_budget_uses_measured_training_stream_average(self):
        steps = []
        for text in ('Hello world!\n' * 40, 'Unseen 世界 🙂\n' * 40):
            with self.subTest(text=text[:20]):
                data = torch.tensor(self.tokenizer.encode(text))
                statistics = train.source_statistics(data, self.tokenizer)
                average = len(text.encode('utf-8')) / len(data)
                self.assertEqual(statistics['average_bytes_per_token'], average)
                config = Config(tokenizer_type='bpe', steps=999, validate_every=17)
                train.configure_training_budget(config, statistics['average_bytes_per_token'])
                bytes_per_step = config.batch_size * config.context * average
                self.assertEqual(config.steps, round(81_920_000 / bytes_per_step))
                self.assertEqual(config.validate_every, max(1, config.steps // 10))
                self.assertLessEqual(
                    abs(config.steps * bytes_per_step - 81_920_000),
                    bytes_per_step / 2,
                )
                steps.append(config.steps)
        self.assertNotEqual(*steps)

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
        training_text = ['Hello hello world!\n' * 20] * 2
        validation_text = ['Unseen 世界 🙂 validation!\n' * 10] * 2
        expected_tokenizer = BPETokenizer.train(training_text, 300, 2)
        average = sum(len(doc.encode('utf-8')) for doc in training_text) / len(train.encode_documents(training_text, expected_tokenizer))
        budget = round(2.4 * 2 * 8 * average)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = self.small_config(tokenizer_type='bpe', bpe_vocab_size=300, target_training_bytes=budget, steps=99, validate_every=17, validation_batches=1, final_validation_batches=1)
            repository = ModelRepository(directory)
            with patch('train.load_source', side_effect=[training_text, validation_text]), patch('train.create_tokenizer', wraps=create_tokenizer) as factory, patch.object(torch.cuda, 'is_available', return_value=False):
                train.train(config, repository, time.perf_counter())
            factory.assert_called_once_with(config, training_text)
            model, checkpoint = repository.load(repository.last_filename, 'cpu')
            self.assertEqual(checkpoint['step'], 2)
            self.assertEqual(checkpoint['run_config']['steps'], 2)
            self.assertEqual(checkpoint['run_config']['validate_every'], 1)
            self.assertEqual(checkpoint['run_config']['target_training_bytes'], budget)
            self.assertEqual(checkpoint['data_metadata']['training']['average_bytes_per_token'], average)
            tokenizer = tokenizer_from_metadata(checkpoint['tokenizer'])
            self.assertEqual(checkpoint['data_metadata']['validation']['raw_utf8_bytes'], sum(len(doc.encode('utf-8')) for doc in validation_text))
            for document in validation_text:
                self.assertEqual(tokenizer.decode(tokenizer.encode(document)), document)
            result = generate(model, tokenizer, 'Hello', 5)
            self.assertGreaterEqual(result[2], 0)
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
