import unittest
import torch
from src.feature_selection import GatedMLP, RankingConfig, cubic_scale, fit_ranking
from src.feature_scores import gate_score_moments, score_from_moments, classification_loss, importance_scaling


class ScoreTests(unittest.TestCase):
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
