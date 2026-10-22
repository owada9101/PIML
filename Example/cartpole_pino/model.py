"""1D FNO accelerations followed by differentiable trapezoidal integration."""

import torch
from torch import nn
from torch.nn import functional as F

from .dynamics import Parameters


class SpectralLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        # Store real/imag separately so optimizers and checkpoints remain ordinary tensors.
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2) / width)

    def forward(self, value):
        transformed = torch.fft.rfft(value)
        out = torch.zeros_like(transformed)
        modes = min(self.modes, transformed.shape[-1])
        weight = torch.view_as_complex(self.weight.contiguous())[:, :, :modes]
        out[..., :modes] = torch.einsum("bim,iom->bom", transformed[..., :modes], weight)
        return torch.fft.irfft(out, n=value.shape[-1])


def cumulative_trapezoid(value, dt):
    increments = (value[:, 1:] + value[:, :-1]) * (dt / 2)
    return torch.cat([torch.zeros_like(value[:, :1]), increments.cumsum(dim=1)], dim=1)


class PINO(nn.Module):
    def __init__(self, width=32, modes=12, layers=3, horizon=0.4, p=Parameters()):
        super().__init__()
        self.horizon = horizon
        self.p = p
        self.config = dict(width=width, modes=modes, layers=layers, horizon=horizon)
        self.register_buffer("scales", torch.tensor(p.scales, dtype=torch.float32))
        self.register_buffer("accel_scales", torch.tensor([p.g, p.g / p.l]))
        self.lift = nn.Conv1d(6, width, 1)
        self.spectral = nn.ModuleList([SpectralLayer(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(),
                                  nn.Conv1d(width, 2, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, initial, forces):
        """Return states [B,N,4] and accelerations [B,N,2] in SI units."""
        n = forces.shape[1]
        t = torch.linspace(0, 1, n, device=forces.device, dtype=forces.dtype)
        conditions = (initial / self.scales)[:, :, None].expand(-1, -1, n)
        inputs = torch.cat([conditions, (forces / ((self.p.M + self.p.m) * self.p.g))[:, None],
                            t[None, None].expand(initial.shape[0], 1, -1)], dim=1)
        value = self.lift(inputs)
        # Zero padding mitigates circular wrap-around; no periodic time BC is imposed.
        padding = max(1, n // 4)
        value = F.pad(value, (0, padding))
        for spectral, local in zip(self.spectral, self.local):
            value = F.gelu(spectral(value) + local(value))
        acceleration = self.head(value)[..., :n].transpose(1, 2) * self.accel_scales
        dt = self.horizon / (n - 1)
        velocity = initial[:, None, [1, 3]] + cumulative_trapezoid(acceleration, dt)
        position = initial[:, None, [0, 2]] + cumulative_trapezoid(velocity, dt)
        states = torch.stack([position[..., 0], velocity[..., 0],
                              position[..., 1], velocity[..., 1]], dim=-1)
        return states, acceleration

    def residual(self, states, acceleration, forces):
        """Nonlinear acceleration-form residual, normalized to dimensionless units."""
        theta, omega = states[..., 2], states[..., 3]
        s, c = theta.sin(), theta.cos()
        p = self.p
        xdd = (forces + p.m * p.g * s * c - p.m * p.l * s * omega.square())
        xdd = xdd / (p.M + p.m * s.square())
        alpha = (c * xdd + p.g * s) / p.l
        target = torch.stack([xdd, alpha], dim=-1)
        return (acceleration - target) / self.accel_scales
