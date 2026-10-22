"""Point mass on a massless rod; theta=0 upright, positive to the left.

State order throughout this example: [x, x_dot, theta, theta_dot].
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class Parameters:
    M: float = 1.0
    m: float = 0.1
    l: float = 0.5
    g: float = 9.81

    @property
    def scales(self):
        return np.array([self.l, np.sqrt(self.g * self.l), 1.0,
                         np.sqrt(self.g / self.l)])


def rhs(state, force, p=Parameters()):
    """Vectorized nonlinear ODE, with arbitrary leading batch dimensions."""
    _, velocity, theta, omega = np.moveaxis(np.asarray(state), -1, 0)
    s, c = np.sin(theta), np.cos(theta)
    acceleration = (force + p.m * p.g * s * c - p.m * p.l * s * omega**2)
    acceleration = acceleration / (p.M + p.m * s**2)
    alpha = (c * acceleration + p.g * s) / p.l
    return np.stack([velocity, acceleration, omega, alpha], axis=-1)


def rk4_step(state, force_start, force_end, dt, p=Parameters(), substeps=4):
    """RK4 with linearly varying force over a single interval."""
    if dt <= 0 or substeps < 1:
        raise ValueError("dt and substeps must be positive")
    state = np.asarray(state, dtype=np.float64).copy()
    h = dt / substeps
    for j in range(substeps):
        u0 = force_start + (force_end - force_start) * j / substeps
        u1 = force_start + (force_end - force_start) * (j + 1) / substeps
        um = (u0 + u1) / 2
        k1 = rhs(state, u0, p)
        k2 = rhs(state + h * k1 / 2, um, p)
        k3 = rhs(state + h * k2 / 2, um, p)
        k4 = rhs(state + h * k3, u1, p)
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state


def simulate(initial, forces, times, p=Parameters()):
    """Integrate sampled forces; the last force axis is time."""
    if forces.shape[-1] != len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("forces must match a strictly increasing time grid")
    trajectory = [np.asarray(initial, dtype=np.float64)]
    for i, dt in enumerate(np.diff(times)):
        trajectory.append(rk4_step(trajectory[-1], forces[..., i],
                                  forces[..., i + 1], dt, p))
    return np.stack(trajectory, axis=-2)


def energy(state, p=Parameters()):
    _, v, theta, omega = np.moveaxis(np.asarray(state), -1, 0)
    return (0.5 * (p.M + p.m) * v**2
            - p.m * p.l * v * omega * np.cos(theta)
            + 0.5 * p.m * p.l**2 * omega**2
            + p.m * p.g * p.l * np.cos(theta))


def lqr(p=Parameters()):
    """Continuous-time local LQR; returns K, terminal matrix P, Q, R."""
    A = np.array([[0, 1, 0, 0], [0, 0, p.m * p.g / p.M, 0],
                  [0, 0, 0, 1], [0, 0, (p.M + p.m) * p.g / (p.M * p.l), 0]])
    B = np.array([[0], [1 / p.M], [0], [1 / (p.M * p.l)]])
    Q = np.diag([4.0, 1.0, 80.0, 2.0])
    R = np.array([[0.1]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K[0], P, Q, float(R[0, 0])


def make_dataset(count, points=65, horizon=0.4, knots=9, seed=0,
                 force_limit=10.0, p=Parameters()):
    """Independent initial states and piecewise-linear force functions."""
    if points < 3 or knots < 2 or (points - 1) % (knots - 1):
        raise ValueError("grid must include every force knot")
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-1, 1, (count, 4)) * [0.4, 1.0, 0.3, 1.5]
    controls = rng.uniform(-force_limit, force_limit, (count, knots))
    times = np.linspace(0, horizon, points)
    forces = np.stack([np.interp(times, np.linspace(0, horizon, knots), u)
                       for u in controls])
    states = simulate(initial, forces, times, p)
    return initial, forces, states
