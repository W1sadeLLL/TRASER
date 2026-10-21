"""Small deterministic unit tests, never empirical paper reproduction."""

import unittest
import numpy as np
import torch
from traser.dynamics.initialization import initialize
from traser.dynamics.operators import directions, bcm_step
from traser.experiments.interventions import weighted_hk
from traser.assimilation.transport import effective_strength


class NumericalContract(unittest.TestCase):
    def test_angular_quadrature_has_four_orthogonal_bases(self):
        theta = directions(2026101903)
        self.assertEqual(theta.shape, (64, 256))
        for start in range(0, 256, 64):
            np.testing.assert_allclose(
                theta[:, start : start + 64].T @ theta[:, start : start + 64],
                np.eye(64),
                atol=2e-15,
            )

    def test_hk_hard_radius_includes_boundary_and_self(self):
        x = torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
        expected = torch.tensor([[0.5], [0.5], [3.0]], dtype=torch.float64)
        torch.testing.assert_close(bcm_step(x, 1.0, 1.0, 2), expected)
        torch.testing.assert_close(
            weighted_hk(x, 1.0, 1.0, torch.ones(3, dtype=x.dtype), 2), expected
        )

    def test_source_silence_preserves_self_edge(self):
        x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        y = weighted_hk(x, 1.0, 2.0, torch.zeros(2, dtype=x.dtype), 2)
        torch.testing.assert_close(y, x)

    def test_initialization_keeps_particle_budget_with_tiny_support(self):
        x = np.arange(12, dtype=float).reshape(3, 4)
        a, ix = initialize(x, 8, 11)
        b, jx = initialize(x, 8, 11)
        self.assertEqual(a.shape, (8, 4))
        np.testing.assert_array_equal(ix, jx)
        counts = np.bincount(ix, minlength=3)
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_strength_is_causal_and_bounded(self):
        spec = {"strength": 2.0, "count_power": 1.0, "innovation_response": 2.0}
        reference = {"count": 100.0, "innovation": 0.1}
        self.assertEqual(effective_strength(spec, 1, 0, reference), 32.0)
        self.assertEqual(effective_strength(spec, 100000, 100, reference), 2 / 32)


if __name__ == "__main__":
    unittest.main()
