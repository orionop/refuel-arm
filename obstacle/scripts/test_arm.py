#!/usr/bin/env python3
"""
Smoke test for the KR6 R700 in obstacle_test.sdf.

Sequence:
  1. Hold initial pose (J2 = -pi/2) for 1s.
  2. Sweep each joint individually through a small range and back.
  3. Return to initial pose.

Usage:
  Terminal A:  ./obstacle/launch.sh
  Terminal B:  python3 obstacle/scripts/test_arm.py
"""
import math
import time

from joint_control import send_q, JOINT_NAMES

Q_INIT = [0.0, -math.pi / 2, 0.0, 0.0, 0.0, 0.0]


def linspace(a: float, b: float, n: int):
    if n <= 1:
        return [b]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def sweep_joint(idx: int, amplitude: float = 0.6, n_steps: int = 40,
                step_dt: float = 0.04) -> None:
    """Move one joint from init -> +amp -> -amp -> init."""
    base = list(Q_INIT)
    print(f"  Sweeping {JOINT_NAMES[idx]} ±{amplitude:.2f} rad")
    targets = (
        linspace(0.0, +amplitude, n_steps)
        + linspace(+amplitude, -amplitude, 2 * n_steps)
        + linspace(-amplitude, 0.0, n_steps)
    )
    for delta in targets:
        q = list(base)
        q[idx] = base[idx] + delta
        send_q(q)
        time.sleep(step_dt)


def main() -> None:
    print("[test_arm] Settling at initial pose")
    for _ in range(20):
        send_q(Q_INIT)
        time.sleep(0.05)
    time.sleep(1.0)

    for j in range(6):
        sweep_joint(j, amplitude=0.6 if j < 3 else 1.0)
        time.sleep(0.3)

    print("[test_arm] Returning to initial pose")
    for _ in range(20):
        send_q(Q_INIT)
        time.sleep(0.05)
    print("[test_arm] Done")


if __name__ == "__main__":
    main()
