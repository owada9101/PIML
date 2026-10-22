"""Train at fine physics/coarse data grids; select a checkpoint by validation only."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dynamics import Parameters, make_dataset
from .model import PINO


def tensor_dataset(count, points, horizon, seed):
    return tuple(torch.tensor(a, dtype=torch.float32)
                 for a in make_dataset(count, points=points, horizon=horizon, seed=seed))


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PINO(**checkpoint["config"], p=Parameters(**checkpoint["parameters"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("Example/cartpole_pino/outputs/pino"))
    args = parser.parse_args()
    if args.steps < 1 or args.train_samples < 1 or args.batch_size < 1 or args.physics_weight < 0:
        parser.error("steps/samples/batch must be positive; physics weight must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = PINO()
    train = tensor_dataset(args.train_samples, 65, model.horizon, args.seed)
    validation = tensor_dataset(64, 65, model.horizon, args.seed + 10000)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-4)
    history, best = [], float("inf")
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        indices = rng.integers(0, args.train_samples, args.batch_size)
        initial, forces, target = (a[indices] for a in train)
        predicted, acceleration = model(initial, forces)
        data_loss = ((predicted[:, ::4] - target[:, ::4]) / model.scales).square().mean()
        physics_loss = model.residual(predicted, acceleration, forces).square().mean()
        loss = data_loss + args.physics_weight * physics_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val_prediction, _ = model(validation[0], validation[1])
                val_loss = ((val_prediction - validation[2]) / model.scales).square().mean().item()
            row = dict(step=step, data=data_loss.item(), physics=physics_loss.item(), validation=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_loss < best:
                best = val_loss
                torch.save(dict(model=model.state_dict(), config=model.config,
                                parameters=asdict(model.p), seed=args.seed,
                                physics_weight=args.physics_weight, best_step=step), args.out / "model.pt")
    elapsed = time.perf_counter() - start
    model, checkpoint = load_model(args.out / "model.pt")
    # A separate force/initial-state draw is used once, after checkpoint selection.
    test = tensor_dataset(64, 65, model.horizon, args.seed + 20000)
    fine = tensor_dataset(64, 129, model.horizon, args.seed + 20000)
    with torch.no_grad():
        prediction, acceleration = model(test[0], test[1])
        fine_prediction, _ = model(fine[0], fine[1])
        metrics = dict(best_step=checkpoint["best_step"], training_seconds=elapsed,
                       test_rmse_si=(prediction - test[2]).square().mean(dim=(0, 1)).sqrt().tolist(),
                       test_scaled_rmse=(((prediction - test[2]) / model.scales).square().mean().sqrt().item()),
                       test_physics_rms=model.residual(prediction, acceleration, test[1]).square().mean().sqrt().item(),
                       fine_scaled_rmse=(((fine_prediction - fine[2]) / model.scales).square().mean().sqrt().item()))
    (args.out / "training.json").write_text(json.dumps(dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        state_order=["x_m", "velocity_m_s", "theta_rad", "omega_rad_s"],
        torch_version=str(torch.__version__), metrics=metrics, history=history), indent=2) + "\n")
    np.savez_compressed(args.out / "prediction.npz", initial=test[0].numpy(), force=test[1].numpy(),
                        target=test[2].numpy(), predicted=prediction.numpy(), horizon=model.horizon)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    times = np.linspace(0, model.horizon, 65)
    for j, (ax, label) in enumerate(zip(axes.flat, ["x [m]", "velocity [m/s]", "theta [rad]", "omega [rad/s]"])):
        ax.plot(times, test[2][0, :, j], label="RK4 reference")
        ax.plot(times, prediction[0, :, j], "--", label="Learned operator")
        ax.set(xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.savefig(args.out / "prediction.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
