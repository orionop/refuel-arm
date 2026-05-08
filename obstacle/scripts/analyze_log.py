#!/usr/bin/env python3
"""Post-process a safety_node CSV and compute the same metrics as the offline
benchmark (`obstacle/scripts/benchmark.py`) so we can compare Gazebo runs to
the pure-python sim.

Usage:
    python3 obstacle/scripts/analyze_log.py path/to/run.csv [--danger 0.5]
    python3 obstacle/scripts/analyze_log.py path/to/run.csv --plot

Pass multiple files to get a side-by-side comparison table:
    python3 obstacle/scripts/analyze_log.py logs/threshold_*.csv logs/hocbf_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from safety.harness.metrics import compute_metrics


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def metrics_from_df(df: pd.DataFrame, *, scenario_name: str, method_name: str,
                    danger_distance: float, expect_collision: bool):
    times = df["t"].to_numpy()
    ee = df[["ee_x", "ee_y", "ee_z"]].to_numpy()
    obs = df[["obs_x", "obs_y", "obs_z"]].to_numpy()
    # Obstacle radius isn't in the CSV (the simulator decides) — caller can
    # pass it via `--radius`. Default 0.10 m matches scenarios.py.
    obs_r = np.full(len(df), 0.10)
    qdot_cmd = df[[f"qdot{i}" for i in range(1, 7)]].to_numpy()
    # nominal qdot also isn't in the CSV; fall back to "first row of qdot
    # before the obstacle entered danger zone" as a proxy.
    seps = np.linalg.norm(ee - obs, axis=1) - obs_r
    pre_danger = seps > danger_distance
    if np.any(pre_danger):
        nominal = qdot_cmd[np.argmax(pre_danger)].copy()
    else:
        nominal = np.zeros(6)
    qdot_nom = np.tile(nominal, (len(df), 1))
    return compute_metrics(
        scenario_name=scenario_name,
        method_name=method_name,
        times=times,
        ee_positions=ee,
        obstacle_positions=obs,
        obstacle_radii=obs_r,
        qdot_cmds=qdot_cmd,
        qdot_nominals=qdot_nom,
        danger_distance=danger_distance,
        expect_collision=expect_collision,
    )


def _name_from_path(p: Path) -> tuple[str, str]:
    """Return (method, scenario_tag) inferred from filename
    `<method>_<run_tag>_<stamp>.csv`."""
    stem = p.stem
    parts = stem.split("_")
    method = parts[0]
    tag = "_".join(parts[1:-2]) if len(parts) >= 4 else "run"
    return method, tag


def _print_table(rows):
    header = (f"{'file':<35s} {'method':<10s} {'scenario':<14s} "
              f"{'min_sep':>8s} {'coll':>5s} {'reac_t':>8s} "
              f"{'dev_l2':>8s} {'tvv':>8s} {'pk_jerk':>10s}")
    print(header)
    print("-" * len(header))
    for path, m in rows:
        rt = f"{m.reaction_time:.3f}" if not np.isnan(m.reaction_time) else "  -  "
        print(f"{path.name[:34]:<35s} {m.method:<10s} {m.scenario:<14s} "
              f"{m.min_separation:>8.3f} {str(m.collision):>5s} "
              f"{rt:>8s} {m.deviation_l2:>8.2f} "
              f"{m.total_vel_variation:>8.2f} {m.peak_jerk:>10.1f}")


def _maybe_plot(df: pd.DataFrame, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seps = np.linalg.norm(df[["ee_x", "ee_y", "ee_z"]].to_numpy()
                          - df[["obs_x", "obs_y", "obs_z"]].to_numpy(), axis=1) - 0.10
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax[0].plot(df["t"], seps); ax[0].axhline(0, color="r", lw=0.5)
    ax[0].set_ylabel("sep − r [m]"); ax[0].set_title(label)
    qdot = df[[f"qdot{i}" for i in range(1, 7)]].to_numpy()
    ax[1].plot(df["t"], np.linalg.norm(qdot, axis=1))
    ax[1].set_ylabel("‖qdot‖"); ax[1].set_xlabel("t [s]")
    out = Path(label).with_suffix(".png")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  plot: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--danger", type=float, default=0.5)
    ap.add_argument("--no-collision", action="store_true",
                    help="treat as a passing/false-positive scenario")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rows = []
    for path in args.csv:
        df = _load(path)
        method, tag = _name_from_path(path)
        m = metrics_from_df(
            df,
            scenario_name=tag, method_name=method,
            danger_distance=args.danger,
            expect_collision=not args.no_collision,
        )
        rows.append((path, m))
        if args.plot:
            _maybe_plot(df, label=path.stem)

    _print_table(rows)


if __name__ == "__main__":
    main()
