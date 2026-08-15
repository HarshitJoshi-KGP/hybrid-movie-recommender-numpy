# scripts/tune.py
"""
Lightweight hyperparameter search + experiment tracking.

Runs a grid over (n_factors, lr, reg) for the from-scratch
MatrixFactorizationSGD model, evaluates each config on a held-out
time-based validation split, and appends one row per run to
experiments/experiment_log.csv.

This intentionally does NOT pull in MLflow/W&B -- for a project this size,
an append-only CSV with one row per run (config + metrics + wall-clock
time, timestamped) is a full audit trail of every experiment and needs
zero extra infrastructure. It's a genuinely useful starting point, not a
toy: `pd.read_csv("experiments/experiment_log.csv")` in a notebook is
enough to compare runs, and the format is easy to graduate to a real
tracker later if the project grows.

Usage:
    python -m scripts.tune                  # run the default grid
    python -m scripts.tune --epochs 15      # override epochs for every config

Each run trains on the same 80% time-based split used in src/evaluate.py
and reports RMSE/MAE on the remaining 20%, so numbers here are directly
comparable to the "Evaluation" table in the README.
"""
import argparse
import csv
import time
from itertools import product
from pathlib import Path

from data.preprocess import load_data
from src.evaluate import compute_rating_metrics, time_based_split
from src.model import MatrixFactorizationSGD

LOG_PATH = Path("experiments/experiment_log.csv")

# Small, fast-to-run default grid. Widen this once you're happy with the
# runtime per config -- e.g. n_factors: [10, 20, 40], lr: [0.002, 0.005,
# 0.01, 0.02], reg: [0.01, 0.02, 0.05].
GRID = {
    "n_factors": [10, 20],
    "lr": [0.005, 0.01],
    "reg": [0.02],
}


def run_grid(n_epochs: int, grid: dict = GRID, log_path: Path = LOG_PATH):
    ratings, _ = load_data()
    train, val = time_based_split(ratings, test_ratio=0.2)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    keys = list(grid.keys())
    configs = [dict(zip(keys, values)) for values in product(*grid.values())]
    print(f"Running {len(configs)} configs x {n_epochs} epochs each...")

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["timestamp", "n_factors", "lr", "reg", "n_epochs", "rmse", "mae", "train_seconds"]
            )

        best = None
        for config in configs:
            print(f"\n--- config: {config} ---")
            model = MatrixFactorizationSGD(n_epochs=n_epochs, **config)

            t0 = time.time()
            model.fit(train)
            elapsed = time.time() - t0

            y_true = val["rating"].to_numpy()
            y_pred = [model.predict(u, m) for u, m in zip(val["user_id"], val["movie_id"])]
            metrics = compute_rating_metrics(y_true, y_pred)

            row = [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                config["n_factors"],
                config["lr"],
                config["reg"],
                n_epochs,
                round(metrics["RMSE"], 4),
                round(metrics["MAE"], 4),
                round(elapsed, 1),
            ]
            writer.writerow(row)
            f.flush()

            print(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  ({elapsed:.1f}s)")
            if best is None or metrics["RMSE"] < best[1]:
                best = (config, metrics["RMSE"])

    print(f"\nBest config on validation RMSE: {best[0]} -> RMSE={best[1]:.4f}")
    print(f"Full log: {log_path}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per config (default: 10)")
    args = parser.parse_args()
    run_grid(n_epochs=args.epochs)
