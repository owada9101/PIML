"""Held-out and shifted-input evaluation of an already trained, paired trial."""

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .control import OperatorMPC, animate, rollout
from .dynamics import Parameters, make_dataset, simulate
from .model import PINO
from .train import load_model


QUANTIZATION = np.array([0.01, 0.05, np.deg2rad(1), 0.1])


def datasets():
    # These draws are fixed across all models and never used for model selection.
    initial, force, target = make_dataset(128, seed=90210)
    cases = {"in_distribution": (initial, force, target)}
    rng = np.random.default_rng(90211)
    times = np.linspace(0, 0.4, 65)
    angle_initial = initial.copy()
    angle_initial[:, 2] = rng.choice([-1, 1], 128) * np.deg2rad(rng.uniform(20, 25, 128))
    cases["angle_20_to_25_deg"] = (angle_initial, force, simulate(angle_initial, force, times))
    knots = rng.choice([-1, 1], (128, 9)) * rng.uniform(10, 16, (128, 9))
    strong = np.stack([np.interp(times, np.linspace(0, 0.4, 9), k) for k in knots])
    cases["force_knots_10_to_16_N"] = (initial, strong, simulate(initial, strong, times))
    # A rectangular pulse resolved on the input grid: edges ramp over one 6.25ms cell.
    pulse = np.zeros_like(force)
    for row in pulse:
        start = int(rng.integers(8, 32))
        end = int(rng.integers(start + 4, 57))
        row[start:end] = rng.uniform(-10, 10)
    cases["pulse_shape"] = (initial, pulse, simulate(initial, pulse, times))
    coarse_initial = np.round(initial / QUANTIZATION) * QUANTIZATION
    cases["quantized_initial_state"] = (coarse_initial, force, target)
    return cases


def predict_metrics(model, data):
    initial, force, target = [torch.tensor(a, dtype=torch.float32) for a in data]
    with torch.no_grad():
        prediction, acceleration = model(initial, force)
        error = prediction - target
        metrics = dict(scaled_rmse=float((error / model.scales).square().mean().sqrt()),
                       rmse_si=error.square().mean((0, 1)).sqrt().tolist(),
                       physics_rms=float(model.residual(prediction, acceleration, force).square().mean().sqrt()))
    return metrics, prediction.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = args.trial / "evaluation"
    out.mkdir(exist_ok=False)
    cases = datasets()
    for case, (initial, force, target) in cases.items():
        np.savez_compressed(out / f"data_{case}.npz", initial=initial, force=force, target=target)
    report = dict(environment=dict(python=platform.python_version(), torch=str(torch.__version__),
                                   platform=platform.platform(), threads=args.threads,
                                   device="cpu", cuda_available=torch.cuda.is_available()),
                  prediction=[], inference=[], controls=[], grid=[])
    torch.manual_seed(0)
    baseline = PINO()
    report["untrained"] = predict_metrics(baseline, cases["in_distribution"])[0]
    models, trainings = {}, {}
    for seed in range(3):
        for kind in ["pino", "fno"]:
            model, checkpoint = load_model(args.trial / f"{kind}_seed{seed}" / "model.pt")
            models[kind, seed] = model
            trainings[kind, seed] = json.loads((args.trial / f"{kind}_seed{seed}" / "training.json").read_text())
            for case, data in cases.items():
                metrics, prediction = predict_metrics(model, data)
                report["prediction"].append(dict(kind=kind, seed=seed, case=case, **metrics))
                np.savez_compressed(out / f"pred_{kind}_{seed}_{case}.npz", predicted=prediction)
            initial, force, _ = cases["in_distribution"]
            state_t = torch.tensor(initial[:1], dtype=torch.float32)
            force_t = torch.tensor(force[:1], dtype=torch.float32)
            with torch.no_grad():
                for _ in range(10):
                    model(state_t, force_t)
                elapsed = []
                for _ in range(100):
                    start = time.perf_counter()
                    model(state_t, force_t)
                    elapsed.append((time.perf_counter() - start) * 1000)
            report["inference"].append(dict(kind=kind, seed=seed, median_ms=float(np.median(elapsed)),
                                            p95_ms=float(np.quantile(elapsed, 0.95)), max_ms=max(elapsed)))
    print("Prediction evaluation complete", flush=True)

    def run(kind, seed, angle=10, disturbance=2, coarse=False, bias=0, group="controls", label="nominal"):
        model = models.get((kind, seed))
        mpc = OperatorMPC(model) if model is not None else None
        name = kind + "_mpc" if mpc is not None else kind
        result, metrics = rollout(name, [0.2, 0, np.deg2rad(angle), 0], 5, Parameters(), mpc,
                                  disturbance=disturbance, quantization=QUANTIZATION if coarse else None,
                                  observe_disturbance=coarse, disturbance_bias=bias)
        metrics.update(kind=kind, seed=seed, initial_angle_deg=angle, disturbance_N=disturbance,
                       coarse_observation=coarse, disturbance_bias_N=bias, case=label)
        report[group].append(metrics)
        np.savez_compressed(out / f"{group}_{label}_{kind}_{seed}_a{angle}_d{disturbance}_b{bias}.npz", **result)
        (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: metrics[k] for k in ["kind", "seed", "case", "initial_angle_deg",
                                                  "disturbance_N", "balanced_entire_run", "settled_at_end"]}), flush=True)
        return result

    nominal = {}
    for kind in ["uncontrolled", "lqr"]:
        nominal[kind] = run(kind, 0)
    for seed in range(3):
        for kind in ["pino", "fno"]:
            result = run(kind, seed)
            if seed == 0:
                nominal[kind] = result
    for bias in [-1, 0, 1]:
        for kind in ["pino", "fno", "lqr"]:
            if kind == "lqr" and bias != 0:
                continue  # LQR only consumes the quantized state, not the estimate.
            run(kind, 0, coarse=True, bias=bias, label="coarse")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for kind, result in nominal.items():
        axes[0].plot(result["times"], np.rad2deg(result["states"][:, 2]), label=kind)
        axes[1].plot(result["times"], result["states"][:, 0], label=kind)
        axes[2].plot(result["times"][:-1], result["forces"][:, 0], label=kind)
    for ax, ylabel in zip(axes, ["Angle [deg]", "Cart position [m]", "Control force [N]"]):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.axvspan(1.5, 1.6, color="orange", alpha=0.2)
    axes[0].legend()
    axes[0].set_title("Seed 0: initial tilt 10 deg, unknown 2 N disturbance")
    axes[-1].set_xlabel("Time [s]")
    fig.savefig(out / "control_comparison.png", dpi=160)
    plt.close(fig)
    animate(nominal["pino"], out / "pino_control.gif", Parameters(), "PINO + MPC (seed 0)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for (kind, seed), training in trainings.items():
        history = training["history"]
        axes[0].semilogy([r["step"] for r in history], [r["validation"] ** 0.5 for r in history],
                         color="#167b8a" if kind == "pino" else "#ce7532", alpha=0.7,
                         label=kind.upper() if seed == 0 else None)
    axes[0].set(xlabel="Training updates", ylabel="Validation scaled RMSE", title="Learning curves: 3 seeds")
    axes[0].legend()
    names = list(cases)
    for kind, offset, color in [("pino", -0.18, "#167b8a"), ("fno", 0.18, "#ce7532")]:
        values = np.array([[r["scaled_rmse"] for r in report["prediction"] if r["kind"] == kind and r["case"] == case] for case in names])
        axes[1].bar(np.arange(len(names)) + offset, values.mean(1), width=0.36,
                    yerr=values.std(1), color=color, label=kind.upper(), capsize=3)
    axes[1].set_xticks(range(len(names)), ["ID", "Angle OOD", "Force OOD", "Pulse OOD", "Coarse state"])
    axes[1].set(ylabel="Scaled RMSE (mean +/- population SD)", title="128 shared held-out trajectories per case")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "learning_and_generalization.png", dpi=160)
    plt.close(fig)

    if not args.skip_grid:
        for angle in [-20, -10, 0, 10, 20]:
            for disturbance in [-6, -3, 0, 3, 6]:
                for kind in ["pino", "fno", "lqr"]:
                    run(kind, 0, angle, disturbance, group="grid", label="grid")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, kind in zip(axes, ["pino", "fno", "lqr"]):
            rows = [r for r in report["grid"] if r["kind"] == kind]
            values = np.array([2 if r["settled_at_end"] else 1 if r["balanced_entire_run"] else 0 for r in rows]).reshape(5, 5)
            ax.imshow(values, vmin=0, vmax=2, cmap="RdYlGn", origin="lower")
            ax.set_xticks(range(5), [-6, -3, 0, 3, 6])
            ax.set_yticks(range(5), [-20, -10, 0, 10, 20])
            ax.set(xlabel="Disturbance [N]", ylabel="Initial angle [deg]", title=kind.upper())
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, ["Fail", "Upright", "Settled"][values[i, j]], ha="center", va="center", fontsize=8)
        fig.savefig(out / "control_grid.png", dpi=160)
        plt.close(fig)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation to {out}", flush=True)


if __name__ == "__main__":
    main()
