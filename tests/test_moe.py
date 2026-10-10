import copy
import io
import unittest

import torch
import torch.nn.functional as F

from tinylm import SparseMoE, TinyLM


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_sparse_outputs_and_gradients_match_dense_reference(self):
        moe = SparseMoE(16, num_experts=8, top_k=2).to(self.device)
        reference = copy.deepcopy(moe)
        x = torch.randn(2, 7, 16, device=self.device, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()

        calls = {}
        handles = [
            expert.register_forward_pre_hook(
                lambda module, args, index=index: calls.__setitem__(
                    index, args[0].detach().clone(),
                )
            )
            for index, expert in enumerate(moe.experts)
        ]
        output, aux = moe(x)
        for handle in handles:
            handle.remove()

        tokens = reference_x.reshape(-1, 16)
        probabilities = reference.router(tokens).softmax(dim=-1)
        weights, selected = probabilities.topk(2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        mask = torch.zeros_like(probabilities).scatter(1, selected, weights)
        # Evaluating every expert is ONLY a correctness oracle in this test.
        all_outputs = torch.stack([expert(tokens) for expert in reference.experts], dim=1)
        expected = (all_outputs * mask.unsqueeze(-1)).sum(dim=1).reshape_as(x)
        fractions = F.one_hot(selected, num_classes=8).float().mean(dim=(0, 1))
        expected_aux = 8 * (fractions * probabilities.mean(dim=0)).sum()

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(aux, expected_aux)
        self.assertEqual(sum(value.shape[0] for value in calls.values()), 2 * 7 * 2)
        for index in range(8):
            positions = torch.where(selected == index)[0]
            if positions.numel():
                torch.testing.assert_close(calls[index], tokens[positions])
            else:
                self.assertNotIn(index, calls)

        (output.square().mean() + 0.01 * aux).backward()
        (expected.square().mean() + 0.01 * expected_aux).backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for actual, dense in zip(moe.parameters(), reference.parameters()):
            if actual.grad is None:
                self.assertTrue(dense.grad is None or torch.count_nonzero(dense.grad) == 0)
            else:
                torch.testing.assert_close(actual.grad, dense.grad)
        self.assertGreater(moe.router.weight.grad.abs().sum().item(), 0)

    def test_unselected_experts_are_never_called(self):
        moe = SparseMoE(8, num_experts=8, top_k=2).to(self.device)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0].fill_(1)
            moe.router.weight[1].fill_(0.5)
        calls = []
        handles = [
            expert.register_forward_pre_hook(
                lambda module, args, index=index: calls.append(index)
            )
            for index, expert in enumerate(moe.experts)
        ]
        output, aux = moe(torch.ones(2, 3, 8, device=self.device))
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [0, 1])
        self.assertEqual(moe.last_expert_counts.tolist(), [6, 6, 0, 0, 0, 0, 0, 0])
        (output.square().mean() + 0.01 * aux).backward()
        self.assertTrue(all(expert.gate.weight.grad is None for expert in moe.experts[2:]))

    def test_recurrence_aux_average_and_checkpoint_roundtrip(self):
        model = TinyLM(dim=16, heads=2, max_context=8, transform_layers=2,
                       recurrent_steps=3, num_experts=4, top_k=2).to(self.device)
        aux_values = []
        handles = [block.mlp.register_forward_hook(
            lambda module, args, result: aux_values.append(result[1])
        ) for block in model.blocks]
        tokens = torch.randint(256, (2, 8), device=self.device)
        logits, aux = model(tokens)
        for handle in handles:
            handle.remove()
        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertEqual(len(aux_values), 6)
        torch.testing.assert_close(aux, torch.stack(aux_values).mean())
        self.assertEqual(model.last_expert_counts.sum(dim=-1).tolist(), [96, 96])
        self.assertFalse(any("last_expert_counts" in key for key in model.state_dict()))

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = copy.deepcopy(model)
        restored.load_state_dict(torch.load(buffer, map_location=self.device))
        restored_logits, restored_aux = restored(tokens)
        torch.testing.assert_close(restored_logits, logits)
        torch.testing.assert_close(restored_aux, aux)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 9, dtype=torch.long, device=self.device))

    def test_routing_configuration(self):
        for num_experts, top_k in [(0, 1), (8, 0), (8, 9)]:
            with self.assertRaises(ValueError):
                SparseMoE(8, num_experts, top_k)
        for top_k in [1, 8]:
            moe = SparseMoE(8, num_experts=8, top_k=top_k).to(self.device)
            output, aux = moe(torch.randn(2, 3, 8, device=self.device))
            self.assertEqual(moe.last_expert_counts.sum().item(), 6 * top_k)
            (output.square().mean() + 0.01 * aux).backward()
            self.assertTrue(torch.isfinite(moe.router.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
