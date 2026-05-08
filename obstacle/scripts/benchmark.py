#!/usr/bin/env python3
"""Offline benchmark CLI — runs all 4 methods against all scenarios in pure
Python (no Gazebo, no ROS2). Useful as a smoke test and for paper numbers.

Run from the repo root:
    python3 obstacle/scripts/benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from safety.harness import all_scenarios, print_metrics_table, run_benchmark
from safety.kinematics import UR5Kinematics
from safety.methods import (
    APFCircularFields,
    DistanceThreshold,
    HOCBFFilter,
    NEOVelocityDamper,
)


def main() -> None:
    q0 = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
    methods = [
        DistanceThreshold(),
        APFCircularFields(),
        NEOVelocityDamper(),
        HOCBFFilter(),
    ]
    scenarios = all_scenarios()

    rows = run_benchmark(methods, scenarios, q0,
                         dt=0.05, kinematics=UR5Kinematics())
    print_metrics_table(rows)


if __name__ == "__main__":
    main()
