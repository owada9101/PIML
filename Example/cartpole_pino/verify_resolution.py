"""Repeat feedback control with half the plant integrator step."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .control import OperatorMPC, rollout
from .dynamics import Parameters, rk4_step
from .train import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.trial / "evaluation"
    local_max = np.zeros(4)
    for pattern in ["controls_*.npz", "grid_*.npz"]:
        for path in out.glob(pattern):
            with np.load(path) as data:
                applied = data["actual_total_forces"]
                fine = rk4_step(data["states"][:-1], applied[:, 0], applied[:, 1], 0.05, substeps=20)
                local_max = np.maximum(local_max, np.abs(fine - data["states"][1:]).max(axis=0))
    checks = dict(local_half_step_max_abs_state_difference=local_max.tolist(), closed_loop=[])
    cases = [(kind, seed, 10, 2, "controls", "nominal") for seed in range(3) for kind in ["pino", "fno"]]
    cases += [(kind, 0, angle, force, "grid", "grid") for angle, force in [(-20, -6), (20, 6)] for kind in ["pino", "fno", "lqr"]]
    metadata = json.loads((out / "evaluation.json").read_text())
    for kind, seed, angle, force, group, label in cases:
        mpc = OperatorMPC(load_model(args.trial / f"{kind}_seed{seed}" / "model.pt")[0]) if kind != "lqr" else None
        result, metrics = rollout(kind + "_mpc" if mpc else kind,
                                  [0.2, 0, np.deg2rad(angle), 0], 5, Parameters(), mpc,
                                  disturbance=force, plant_substeps=20)
        reference = next(r for r in metadata[group] if r["kind"] == kind and r["seed"] == seed
                         and r["initial_angle_deg"] == angle and r["disturbance_N"] == force and r["case"] == label)
        with np.load(out / f"{group}_{label}_{kind}_{seed}_a{angle}_d{force}_b0.npz") as original:
            count = min(len(original["states"]), len(result["states"]))
            error = np.abs(original["states"][:count] - result["states"][:count]).max(axis=0)
        row = dict(kind=kind, seed=seed, angle=angle, disturbance=force,
                   balanced=metrics["balanced_entire_run"], settled=metrics["settled_at_end"],
                   same_outcome=all(metrics[key] == reference[key] for key in ["balanced_entire_run", "settled_at_end"]),
                   max_state_difference=error.tolist())
        checks["closed_loop"].append(row)
        np.savez_compressed(out / f"refined_{kind}_{seed}_a{angle}_d{force}.npz", **result)
        (out / "resolution_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
    assert all(r["same_outcome"] for r in checks["closed_loop"])


if __name__ == "__main__":
    main()
