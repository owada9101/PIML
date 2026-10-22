"""Physical consistency checks, independent of learned accuracy."""

import unittest

import numpy as np
import torch

from .dynamics import Parameters, energy, lqr, rhs, simulate
from .model import PINO


class PhysicsChecks(unittest.TestCase):
    def test_equilibrium_and_force_sign(self):
        np.testing.assert_array_equal(rhs(np.zeros(4), 0), np.zeros(4))
        self.assertGreater(rhs(np.zeros(4), 1)[3], 0)
        self.assertGreater(rhs(np.array([0, 0, 0.1, 0]), 0)[3], 0)

    def test_implicit_equations(self):
        rng = np.random.default_rng(2)
        p = Parameters()
        state = rng.normal(size=(32, 4))
        force = rng.normal(size=32)
        d = rhs(state, force, p)
        theta, omega = state[:, 2], state[:, 3]
        first = ((p.M + p.m) * d[:, 1] - p.m * p.l * np.cos(theta) * d[:, 3]
                 + p.m * p.l * np.sin(theta) * omega**2 - force)
        second = p.l * d[:, 3] - np.cos(theta) * d[:, 1] - p.g * np.sin(theta)
        np.testing.assert_allclose(first, 0, atol=1e-13)
        np.testing.assert_allclose(second, 0, atol=1e-13)

    def test_energy_conservation(self):
        t = np.linspace(0, 3, 601)
        state = simulate(np.array([0, 0.2, 0.1, -0.1]), np.zeros(len(t)), t)
        self.assertLess(np.ptp(energy(state)), 1e-8)

    def test_local_lqr_stability(self):
        eps = 1e-6
        zero = np.zeros(4)
        A = np.stack([(rhs(zero + eps * e, 0) - rhs(zero - eps * e, 0)) / (2 * eps)
                      for e in np.eye(4)], axis=1)
        B = ((rhs(zero, eps) - rhs(zero, -eps)) / (2 * eps))[:, None]
        K, _, _, _ = lqr()
        self.assertLess(np.max(np.linalg.eigvals(A - B @ K[None]).real), 0)

    def test_operator_integral_and_control_gradient(self):
        torch.manual_seed(0)
        model = PINO(width=8, modes=4, layers=1)
        initial = torch.tensor([[0.2, 0.1, 0.15, -0.2]])
        forces = torch.zeros(1, 33, requires_grad=True)
        # Nonzero head weights expose the learned force-to-state gradient.
        torch.nn.init.normal_(model.head[-1].weight, std=0.05)
        states, accelerations = model(initial, forces)
        torch.testing.assert_close(states[:, 0], initial, atol=0, rtol=0)
        dt = model.horizon / 32
        torch.testing.assert_close(states[:, 1:, [1, 3]] - states[:, :-1, [1, 3]],
                                   dt * (accelerations[:, 1:] + accelerations[:, :-1]) / 2)
        states[:, -1].square().sum().backward()
        self.assertTrue(torch.isfinite(forces.grad).all())
        self.assertGreater(forces.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
