"""Use the frozen PINO inside MPC and apply its forces to the nonlinear RK4 plant."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
from torch.nn import functional as F

from .dynamics import Parameters, lqr, rk4_step
from .train import load_model


class OperatorMPC:
    def __init__(self, model, iterations=15, force_limit=10.0):
        self.model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.iterations = iterations
        self.force_limit = force_limit
        self.knots = 9
        self.dt = model.horizon / (self.knots - 1)
        self.previous = torch.zeros(self.knots)
        _, terminal, Q, R = lqr(model.p)
        self.terminal = torch.tensor(terminal, dtype=torch.float32)
        self.Q = torch.tensor(Q.diagonal(), dtype=torch.float32)
        self.R = R

    def action(self, state, disturbance_estimate=0.0):
        # Initial guess is the shifted previous plan, zero on the first call.
        raw = torch.cat([self.previous[1:], self.previous[-1:]]).clone().requires_grad_()
        optimizer = torch.optim.LBFGS([raw], lr=1.0, max_iter=self.iterations,
                                      line_search_fn="strong_wolfe")
        initial = torch.tensor(state, dtype=torch.float32)[None]

        def objective():
            optimizer.zero_grad()
            knots = self.force_limit * raw.tanh()
            forces = F.interpolate(knots[None, None], size=65, mode="linear",
                                   align_corners=True)[:, 0]
            states, _ = self.model(initial, forces + disturbance_estimate)
            stage = (states.square() * self.Q).sum(dim=-1) + self.R * forces.square()
            cost = torch.trapezoid(stage, dx=self.model.horizon / 64).mean()
            terminal = states[:, -1]
            cost = cost + torch.einsum("bi,ij,bj->", terminal, self.terminal, terminal)
            cost.backward()
            return cost

        optimizer.step(objective)
        if not torch.isfinite(raw).all():
            raise RuntimeError("non-finite MPC plan")
        self.previous = raw.detach()
        with torch.no_grad():
            knots = self.force_limit * raw.tanh()
            forces = F.interpolate(knots[None, None], size=65, mode="linear",
                                   align_corners=True)[:, 0]
            states, _ = self.model(initial, forces + disturbance_estimate)
        return knots[:2].detach().numpy().astype(float), states[0, 8].numpy()


def rollout(kind, initial, duration, p, mpc=None, disturbance=2.0,
            quantization=None, observe_disturbance=False, disturbance_bias=0.0, plant_substeps=10):
    dt = mpc.dt if mpc is not None else 0.05
    K, _, Q, R = lqr(p)
    states, forces, elapsed, errors = [np.asarray(initial, dtype=float)], [], [], []
    reason = None
    observed_states, total_forces, predicted_forces = [], [], []
    for i in range(round(duration / dt)):
        t = i * dt
        external = disturbance if 1.5 <= t < 1.6 - 1e-9 else 0.0
        observed = states[-1].copy()
        if quantization is not None:
            observed = np.round(observed / quantization) * quantization
        estimate = external + disturbance_bias if observe_disturbance else 0.0
        observed_states.append(observed)
        start = time.perf_counter()
        if kind.endswith("_mpc"):
            action, predicted = mpc.action(observed, estimate)
            predicted_forces.append((mpc.force_limit * mpc.previous.tanh()).numpy() + estimate)
        elif kind == "lqr":
            action = np.full(2, np.clip(-K @ observed, -10, 10))
        else:
            action = np.zeros(2)
        elapsed.append(time.perf_counter() - start)
        # Nominal runs do not expose the disturbance. Optional coarse runs hold
        # the current estimate across the horizon, without future pulse timing.
        total_forces.append(action + external)
        state = rk4_step(states[-1], action[0] + external, action[1] + external,
                         dt, p, substeps=plant_substeps)
        if kind.endswith("_mpc"):
            errors.append((state - predicted).tolist())
        states.append(state)
        forces.append(action)
        if not np.isfinite(state).all():
            reason = "nonfinite_state"
            break
        if abs(state[2]) > np.deg2rad(30):
            reason = "angle_exceeded_30_degrees"
            break
        if abs(state[0]) > 2.4:
            reason = "cart_exceeded_2.4_meters"
            break
    states, forces = np.array(states), np.array(forces)
    times = np.arange(len(states)) * dt
    stage = np.einsum("bi,ij,bj->b", states[:-1], Q, states[:-1])
    stage += R * np.mean(forces**2, axis=1)
    metrics = dict(controller=kind, simulated_seconds=float(times[-1]), failure=reason,
                   balanced_entire_run=reason is None,
                   settled_at_end=bool(reason is None and abs(states[-1, 0]) < 0.1
                                       and abs(states[-1, 1]) < 0.1
                                       and abs(states[-1, 2]) < np.deg2rad(2)
                                       and abs(states[-1, 3]) < 0.1),
                   max_angle_deg=float(np.rad2deg(abs(states[:, 2])).max()),
                   max_cart_m=float(abs(states[:, 0]).max()),
                   final_state=states[-1].tolist(),
                   mean_planning_ms=float(np.mean(elapsed) * 1000),
                   p95_planning_ms=float(np.quantile(elapsed, 0.95) * 1000),
                   max_planning_ms=float(np.max(elapsed) * 1000),
                   max_force_N=float(abs(forces).max()),
                   accumulated_stage_cost=float(stage.sum() * dt))
    warm = np.asarray(elapsed[1:] if len(elapsed) > 1 else elapsed) * 1000
    metrics.update(first_planning_ms=elapsed[0] * 1000,
                   warm_median_planning_ms=float(np.median(warm)),
                   warm_p95_planning_ms=float(np.quantile(warm, 0.95)),
                   warm_max_planning_ms=float(np.max(warm)),
                   warm_deadline_miss_fraction=float(np.mean(warm > dt * 1000)),
                   observation_outside_training_fraction=float(np.mean(np.any(
                       np.abs(observed_states) > np.array([0.4, 1.0, 0.3, 1.5]), axis=1))),
                   max_actual_total_force_N=float(np.abs(total_forces).max()),
                   max_predicted_total_force_N=float(np.abs(predicted_forces).max()) if predicted_forces else None)
    return dict(times=times, states=states, forces=forces, planning_seconds=np.array(elapsed),
                one_step_error=np.array(errors), observed_states=np.array(observed_states),
                actual_total_forces=np.array(total_forces),
                predicted_total_force_knots=np.array(predicted_forces)), metrics


def animate(result, path, p, controller_label):
    states, times = result["states"], result["times"]
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.set(xlim=(-2.7, 2.7), ylim=(-0.35, 0.85), xlabel="Cart position [m]")
    ax.set_aspect("equal")
    ax.axhline(-0.14, color="gray")
    cart = ax.add_patch(Rectangle((0, -0.1), 0.3, 0.16, color="#5e7385"))
    rod, = ax.plot([], [], lw=3, color="#273747")
    bob = ax.add_patch(Circle((0, p.l), 0.045, color="#d76c38"))
    label = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    def update(i):
        x, _, theta, _ = states[i]
        bx, by = x - p.l * np.sin(theta), p.l * np.cos(theta)
        cart.set_x(x - 0.15)
        rod.set_data([x, bx], [0, by])
        bob.center = (bx, by)
        label.set_text(f"{controller_label}    t = {times[i]:.2f} s    theta = {np.rad2deg(theta):.1f} deg")
        return cart, rod, bob, label

    animation = FuncAnimation(fig, update, frames=len(states), interval=50)
    animation.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("Example/cartpole_pino/outputs/pino/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("Example/cartpole_pino/outputs/control"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--angle-deg", type=float, default=10.0)
    parser.add_argument("--cart-position", type=float, default=0.2)
    parser.add_argument("--disturbance", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()
    if args.duration < 0.05 or args.iterations < 1:
        parser.error("duration must be at least 0.05 seconds and iterations positive")
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint)
    initial = [args.cart_position, 0, np.deg2rad(args.angle_deg), 0]
    mpc = OperatorMPC(model, args.iterations)
    learned_kind = "pino_mpc" if checkpoint["physics_weight"] > 0 else "fno_mpc"
    controller_label = "PINO + MPC" if checkpoint["physics_weight"] > 0 else "FNO + MPC"
    results, metrics = {}, []
    for kind in ["uncontrolled", "lqr", learned_kind]:
        result, summary = rollout(kind, initial, args.duration, model.p, mpc,
                                  disturbance=args.disturbance)
        results[kind] = result
        metrics.append(summary)
        np.savez_compressed(args.out / f"{kind}.npz", **result)
        print(json.dumps(summary), flush=True)
    metadata = dict(initial_state=initial, dt=mpc.dt, horizon=model.horizon,
                    checkpoint=str(args.checkpoint.resolve()), training_seed=checkpoint["seed"],
                    physics_weight=checkpoint["physics_weight"],
                    disturbance=dict(force_N=args.disturbance, start_s=1.5, end_s=1.6),
                    model_fallback=False, metrics=metrics)
    (args.out / "control.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True, sharex=True)
    for kind, result in results.items():
        t, states = result["times"], result["states"]
        axes[0].plot(t, np.rad2deg(states[:, 2]), label=kind)
        axes[1].plot(t, states[:, 0], label=kind)
        axes[2].plot(t[:-1], result["forces"][:, 0], label=kind)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.axvspan(1.5, 1.6, alpha=0.1, color="orange")
    axes[0].set(ylabel="Angle [deg]", title=f"Nonlinear plant: {args.angle_deg:g} deg initial tilt + external force")
    axes[0].legend()
    axes[1].set(ylabel="Cart x [m]")
    axes[2].set(ylabel="Force at interval start [N]", xlabel="Time [s]")
    fig.savefig(args.out / "control.png", dpi=160)
    plt.close(fig)
    if not args.no_animation:
        animate(results[learned_kind], args.out / "control.gif", model.p, controller_label)


if __name__ == "__main__":
    main()
