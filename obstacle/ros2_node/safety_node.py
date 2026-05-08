#!/usr/bin/env python3
"""ROS2 safety supervisor node for myCobot 280 in gz sim.

Topology:

    /joint_states                  ── sensor_msgs/JointState   (from gz_ros2_control)
    /obstacle_pose                 ── geometry_msgs/PoseStamped (from obstacle_simulator)
    /arm_controller/               ── trajectory_msgs/JointTrajectory (we publish here)
        commands

This node:
1. Tracks the latest myCobot 280 joint state.
2. Tracks the latest obstacle pose + estimates velocity by finite difference.
3. At a fixed control rate, computes (FK, Jacobian) for the myCobot, builds a
   RobotState + Obstacle, and asks the chosen safety method for a qdot.
4. Publishes the qdot to the arm controller.

The nominal qdot the safety method tries to track is provided as a ROS
parameter (`nominal_qdot`) — replace with whatever your refueling planner
publishes once integrated.

Usage (on Ubuntu 24 + ROS2 Jazzy + gz sim Harmonic with mycobot_ros2):

    ros2 run obstacle safety_node \\
        --ros-args -p method:=hocbf -p control_rate:=50.0
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# Allow running both as a ROS2 package script and directly via python3.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray
except ImportError:                                          # pragma: no cover
    print("[safety_node] rclpy not available — run on ROS2 Jazzy box.")
    raise

from safety.kinematics import MyCobotKinematics
from safety.methods import (
    APFCircularFields,
    DistanceThreshold,
    HOCBFFilter,
    NEOVelocityDamper,
)
from safety.types import Obstacle, RobotState


METHOD_REGISTRY = {
    "threshold": DistanceThreshold,
    "apf": APFCircularFields,
    "neo": NEOVelocityDamper,
    "hocbf": HOCBFFilter,
}

# Joint names exposed by the myCobot 280 in mycobot_ros2 Gazebo simulation.
MYCOBOT_JOINT_ORDER = [
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange",
]


class SafetyNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_node")

        # parameters
        self.declare_parameter("method", "hocbf")
        self.declare_parameter("control_rate", 50.0)
        self.declare_parameter("nominal_qdot", [0.0] * 6)
        self.declare_parameter("controller_topic",
                               "/forward_velocity_controller/commands")
        # CSV logging — written under <log_dir>/<method>_<timestamp>.csv
        self.declare_parameter("log_dir", "")
        self.declare_parameter("run_tag", "")

        method_name = self.get_parameter("method").get_parameter_value().string_value
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"unknown method '{method_name}', "
                             f"choose from {list(METHOD_REGISTRY)}")
        self.method = METHOD_REGISTRY[method_name]()
        self.kin = MyCobotKinematics()

        rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / rate

        nominal = list(self.get_parameter("nominal_qdot").value)
        self.qdot_nominal = np.asarray(nominal, dtype=float)

        controller_topic = self.get_parameter("controller_topic").value

        # ── CSV logger setup ────────────────────────────────────────────
        log_dir = self.get_parameter("log_dir").get_parameter_value().string_value
        run_tag = self.get_parameter("run_tag").get_parameter_value().string_value
        self._log_file = None
        self._log_writer = None
        self._t_run_start = None
        if log_dir:
            ldir = Path(log_dir).expanduser().resolve()
            ldir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            tag = f"_{run_tag}" if run_tag else ""
            path = ldir / f"{method_name}{tag}_{stamp}.csv"
            self._log_file = path.open("w", newline="")
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow([
                "t",
                "q1", "q2", "q3", "q4", "q5", "q6",
                "ee_x", "ee_y", "ee_z",
                "obs_x", "obs_y", "obs_z",
                "obs_vx", "obs_vy", "obs_vz",
                "qdot1", "qdot2", "qdot3", "qdot4", "qdot5", "qdot6",
                "min_d", "safe", "info_json",
            ])
            self.get_logger().info(f"logging to {path}")

        # state caches
        self.q = None         # type: ignore
        self.qdot = None      # type: ignore
        self._joint_index = None

        self._obs_p = None
        self._obs_p_prev = None
        self._obs_t_prev = None
        self._obs_v = np.zeros(3)
        self._obs_radius = 0.10

        # subs + pubs
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states",
                                 self._on_joint_state, sensor_qos)
        self.create_subscription(PoseStamped, "/obstacle_pose",
                                 self._on_obstacle, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, controller_topic, 10)

        self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"safety_node started: method={method_name} rate={rate} Hz "
            f"controller_topic={controller_topic}")

    # ── callbacks ──────────────────────────────────────────────────────
    def _on_joint_state(self, msg: JointState) -> None:
        if self._joint_index is None:
            try:
                self._joint_index = [msg.name.index(j) for j in MYCOBOT_JOINT_ORDER]
            except ValueError as e:
                self.get_logger().warn(f"joint state missing myCobot joint: {e}")
                return
        self.q = np.array([msg.position[i] for i in self._joint_index], dtype=float)
        if msg.velocity and len(msg.velocity) >= len(msg.position):
            self.qdot = np.array([msg.velocity[i] for i in self._joint_index], dtype=float)
        else:
            self.qdot = np.zeros(6)

    def _on_obstacle(self, msg: PoseStamped) -> None:
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                     dtype=float)
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if self._obs_p_prev is not None and self._obs_t_prev is not None:
            dt = t - self._obs_t_prev
            if dt > 1e-3:
                # exponential smoothing of velocity to suppress noise
                inst_v = (p - self._obs_p_prev) / dt
                self._obs_v = 0.7 * self._obs_v + 0.3 * inst_v
        self._obs_p_prev = p
        self._obs_t_prev = t
        self._obs_p = p

    # ── control tick ───────────────────────────────────────────────────
    def _tick(self) -> None:
        if self.q is None:
            return

        R, p_ee = self.kin.fk(self.q)
        J = self.kin.jacobian(self.q)
        state = RobotState(q=self.q, qdot=self.qdot if self.qdot is not None else np.zeros(6),
                           ee_pos=p_ee, ee_R=R, jacobian=J)

        obstacles = []
        if self._obs_p is not None:
            obstacles.append(Obstacle(p=self._obs_p, v=self._obs_v,
                                      radius=self._obs_radius, label="dynamic"))

        out = self.method.step(state, obstacles, self.qdot_nominal)

        msg = Float64MultiArray()
        msg.data = out.qdot_cmd.astype(float).tolist()
        self.cmd_pub.publish(msg)

        if not out.safe:
            self.get_logger().warn(
                f"unsafe: min_d={out.info.get('min_d', float('nan')):.3f} "
                f"info={out.info}")

        # ── log row ─────────────────────────────────────────────────────
        if self._log_writer is not None:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self._t_run_start is None:
                self._t_run_start = now_s
            t_rel = now_s - self._t_run_start
            obs_p = self._obs_p if self._obs_p is not None else np.full(3, np.nan)
            obs_v = self._obs_v if self._obs_p is not None else np.zeros(3)
            min_d = float(out.info.get("min_d", float("nan")))
            row = [
                f"{t_rel:.4f}",
                *[f"{v:.6f}" for v in self.q.tolist()],
                f"{p_ee[0]:.6f}", f"{p_ee[1]:.6f}", f"{p_ee[2]:.6f}",
                f"{obs_p[0]:.6f}", f"{obs_p[1]:.6f}", f"{obs_p[2]:.6f}",
                f"{obs_v[0]:.6f}", f"{obs_v[1]:.6f}", f"{obs_v[2]:.6f}",
                *[f"{v:.6f}" for v in out.qdot_cmd.tolist()],
                f"{min_d:.6f}",
                int(out.safe),
                json.dumps({k: v for k, v in out.info.items() if k != "min_d"}),
            ]
            self._log_writer.writerow(row)
            # flush every ~50 rows so a crash doesn't lose the whole tape
            if int(t_rel * (1.0 / self.dt)) % 50 == 0:
                self._log_file.flush()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    finally:
        if node._log_file is not None:
            node._log_file.flush()
            node._log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
