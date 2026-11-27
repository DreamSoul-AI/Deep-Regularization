import unittest
import os
from pathlib import Path
import subprocess
import torch
from src.feature_selection import GatedMLP, RankingConfig, cubic_scale, fit_ranking
from src.feature_scores import gate_score_moments, score_from_moments, classification_loss, importance_scaling, ranking_score


class ScoreTests(unittest.TestCase):
    def test_weighted_left_input_keeps_the_original_threshold(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        config = RankingConfig(epochs=1, warmup_epochs=0, batch_size=6, dropout=0, feature_dropout=0)
        torch.manual_seed(5)
        initial = GatedMLP(4, config=config)
        norm = initial.layers[1].weight.detach().norm(dim=0)
        threshold = (config.penalty / (norm + 1e-8)).clamp(1e-8, .5) / config.rho
        for left, mode in (("default", "column_norm"), ("weighted_consensus", "gate_only"),
                           ("score", "gate_adaptive"), ("score", "column_norm_gate")):
            _, result = fit_ranking(x, y, mode=mode, seed=5, config=config, prox_input=left)
            gate = result["model"]["gate"]
            value = gate.abs() if left == "score" else gate
            if left == "weighted_consensus" or mode == "column_norm_gate":
                value = value * norm
            shrunk = value.sign() * (value.abs() - threshold).clamp_min(0)
            coefficient = ((config.penalty/config.rho) * shrunk.abs().sum()
                           / (shrunk.norm() + 1e-8).pow(3).clamp_min(1e-10))
            torch.testing.assert_close(result["z"], cubic_scale(coefficient) * shrunk)

    def test_ranking_uses_threshold_formula_and_column_norm_once(self):
        parameters = {"gate": torch.tensor([.5, 2.]), "layers.1.weight": torch.tensor([[3., 0.], [4., 2.]])}
        first, second = torch.tensor([-2., 1.]), torch.tensor([4., 8.])
        torch.testing.assert_close(ranking_score(parameters, "column_norm"), torch.tensor([5., 2.]))
        torch.testing.assert_close(ranking_score(parameters, "column_norm_gate"), torch.tensor([2.5, 4.]))
        torch.testing.assert_close(ranking_score(parameters, "taylor_fisher", first, second), torch.tensor([4., 3.]))
        torch.testing.assert_close(ranking_score(parameters, "column_norm_taylor_fisher", first, second), torch.tensor([20., 6.]))

    def test_without_soft_threshold_training_is_independent_of_score_mode(self):
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        config = RankingConfig(epochs=3, warmup_epochs=1, batch_size=3)
        _, gate = fit_ranking(x, y, mode="gate_only", seed=4, config=config, soft_threshold=False)
        _, weighted = fit_ranking(x, y, mode="column_norm_taylor_second", seed=4,
                                  config=config, soft_threshold=False)
        for name, parameter in gate["model"].items():
            torch.testing.assert_close(parameter, weighted["model"][name], rtol=0, atol=0)
        torch.testing.assert_close(gate["z"], weighted["z"], rtol=0, atol=0)

    def test_legacy_training_matches_fixed_original_parameters(self):
        from experiments.legacy_ranking_reference import COMMIT, load_reference, train_reference
        root = Path(__file__).resolve().parents[1]
        available = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "cat-file", "-e",
             f"{COMMIT}:src/admm_input_group_wrapper.py"], cwd=root,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}, capture_output=True,
        )
        if available.returncode:
            self.skipTest("Fixed historical Git object unavailable; no network fetch in unit tests")
        import numpy as np
        x = np.random.default_rng(0).random((60, 128))
        y = ((x[:, 0]-.5)*(x[:, 1]-.5)>0).astype(int)
        reference, _ = load_reference()
        old = train_reference(reference, x, y, epochs=14, warmup=2)
        _, state = fit_ranking(torch.from_numpy(x), torch.from_numpy(y), mode="column_norm",
                               seed=0, config=RankingConfig(epochs=14, warmup_epochs=2, protocol="legacy"))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, state["model"][key], rtol=0, atol=0)

    def test_per_sample_square_survives_gradient_cancellation(self):
        model = GatedMLP(1, config=RankingConfig(dropout=0, feature_dropout=0)).double()
        # Soft targets straddling sigmoid(logit) give equal opposite BCE gradients.
        model.layers = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Identity()).double()
        with torch.no_grad():
            model.layers[0].weight.fill_(1)
        x = torch.tensor([[1.], [1.]], dtype=torch.double)
        y = torch.sigmoid(torch.ones(2, dtype=torch.double)) + torch.tensor([-.1, .1])
        first, second = gate_score_moments(model, x, y)
        torch.testing.assert_close(first, torch.zeros_like(first), atol=1e-7, rtol=0)
        self.assertGreater(second.item(), .009)
        torch.testing.assert_close(score_from_moments(first, second, "taylor_second"), 0.5 * second)
        self.assertGreater(score_from_moments(first, second, "taylor_fisher").item(), .004)

    def test_vectorized_matches_individual_autograd_and_preserves_state(self):
        for classes in [2, 3]:
            torch.manual_seed(7)
            model = GatedMLP(4, classes).double()
            x = torch.randn(7, 4, dtype=torch.double)
            y = torch.arange(7) % classes
            model.train()
            model.layers[0].eval()  # Mixed flags must survive scoring too.
            flags = [m.training for m in model.modules()]
            model.gate.grad = torch.full_like(model.gate, 3)
            rng = torch.get_rng_state().clone()
            a, b = gate_score_moments(model, x, y, 3)
            self.assertEqual(flags, [m.training for m in model.modules()])
            torch.testing.assert_close(model.gate.grad, torch.full_like(model.gate, 3))
            torch.testing.assert_close(rng, torch.get_rng_state())
            model.eval()
            values = []
            for row, label in zip(x, y):
                loss = classification_loss(model(row[None]), label[None])
                values.append((torch.autograd.grad(loss, model.gate)[0] * model.gate).detach())
            values = torch.stack(values)
            torch.testing.assert_close(a, values.mean(0))
            torch.testing.assert_close(b, values.square().mean(0))
            # Finite difference checks the deletion-direction first derivative.
            original = model.gate.detach().clone()
            eps = 1e-5
            with torch.no_grad():
                model.gate.copy_(original * (1 - eps))
                minus = classification_loss(model(x), y)
                model.gate.copy_(original * (1 + eps))
                plus = classification_loss(model(x), y)
                model.gate.copy_(original)
            torch.testing.assert_close((plus-minus)/(2*eps), a.sum(), atol=1e-7, rtol=1e-5)

    def test_cubic_residual_and_zero_scaling(self):
        d = torch.tensor([0., .1, .4, 1., 1e6, 1e12], dtype=torch.double)
        tau = cubic_scale(d)
        torch.testing.assert_close(tau**3 - tau, d, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(importance_scaling(torch.zeros(3)), torch.ones(3))

    def test_multiclass_admm_roundtrip_and_reproducibility(self):
        torch.manual_seed(1)
        x = torch.randn(18, 5)
        y = torch.arange(18) % 3
        config = RankingConfig(epochs=3, warmup_epochs=1, score_samples=8, hidden_size=4)
        scores, state = fit_ranking(x, y, mode="taylor_fisher", seed=9, config=config)
        duplicate, _ = fit_ranking(x, y, mode="taylor_fisher", seed=9, config=config)
        for name in scores:
            torch.testing.assert_close(scores[name], duplicate[name], rtol=0, atol=0)
        self.assertEqual(len(state["trace"]), 3)
        self.assertTrue(torch.isfinite(state["z"]).all())
        model = GatedMLP(5, 3, config)
        model.load_state_dict(state["model"])
        torch.testing.assert_close(model.gate.abs(), scores["gate"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
