#!/usr/bin/env python3
"""
Minimal gz-transport joint position commander for KR6 R700.

Each joint of the model has a JointPositionController plugin that listens on:
    /model/<model_name>/joint/<joint_name>/0/cmd_pos   (gz.msgs.Double)

This module wraps the `gz topic` CLI so we don't need rclpy or a custom
gz-transport python binding.
"""
import os
import shlex
import subprocess
import time
from typing import Sequence

GZ_BIN = "/opt/homebrew/bin/gz" if os.path.exists("/opt/homebrew/bin/gz") else "gz"

MODEL_NAME = "kr6_r700"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def _cmd_topic(joint: str, model: str = MODEL_NAME) -> str:
    return f"/model/{model}/joint/{joint}/0/cmd_pos"


def send_joint(joint: str, value: float, model: str = MODEL_NAME) -> None:
    """Publish a single Double to one joint's cmd_pos topic."""
    topic = _cmd_topic(joint, model)
    cmd = (
        f"{shlex.quote(GZ_BIN)} topic -t {shlex.quote(topic)} "
        f"-m gz.msgs.Double -p 'data: {float(value)}'"
    )
    subprocess.run(cmd, shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def send_q(q: Sequence[float], model: str = MODEL_NAME) -> None:
    """Publish all six joint positions in parallel."""
    if len(q) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} joint values, got {len(q)}")
    parts = [
        f"{shlex.quote(GZ_BIN)} topic -t {shlex.quote(_cmd_topic(j, model))} "
        f"-m gz.msgs.Double -p 'data: {float(v)}'"
        for j, v in zip(JOINT_NAMES, q)
    ]
    subprocess.run("(" + " & ".join(parts) + " & wait) >/dev/null 2>&1",
                   shell=True, check=False)


def play_trajectory(traj, dt: float = 0.05, model: str = MODEL_NAME) -> None:
    """Send a sequence of joint configurations at fixed step time."""
    for q in traj:
        send_q(q, model=model)
        time.sleep(dt)
