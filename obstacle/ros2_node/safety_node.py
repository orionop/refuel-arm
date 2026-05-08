#!/usr/bin/env python3
"""ROS2 safety supervisor node for UR5 in gz sim.

Topology:

    /joint_states                  ── sensor_msgs/JointState   (from gz_ros2_control)
    /obstacle_pose                 ── geometry_msgs/PoseStamped (from obstacle_simulator)
    /forward_velocity_controller/  ── std_msgs/Float64MultiArray (we publish here)
        commands

This node:
1. Tracks the latest UR5 joint state.
2. Tracks the latest obstacle pose + estimates velocity by finite difference.
3. At a fixed control rate, computes (FK, Jacobian) for the UR5, builds a
   RobotState + Obstacle, and asks the chosen safety method for a qdot.
4. Publishes the qdot to the forward velocity controller.

The nominal qdot the safety method tries to track is provided as a ROS
parameter (`nominal_qdot`) — replace with whatever your refueling planner
publishes once integrated.

Usage (on Ubuntu 24 + ROS2 Jazzy + gz sim Harmonic):

    ros2 run obstacle safety_node \\
        --ros-args -p method:=hocbf -p control_rate:=50.0
"""
from __future__ import annotations

import sys
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

from safety.kinematics import UR5Kinematics
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

# Joint names exposed by the UR5 in Universal_Robots_ROS2_GZ_Simulation.
UR5_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
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

        method_name = self.get_parameter("method").get_parameter_value().string_value
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"unknown method '{method_name}', "
                             f"choose from {list(METHOD_REGISTRY)}")
        self.method = METHOD_REGISTRY[method_name]()
        self.kin = UR5Kinematics()

        rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / rate

        nominal = list(self.get_parameter("nominal_qdot").value)
        self.qdot_nominal = np.asarray(nominal, dtype=float)

        controller_topic = self.get_parameter("controller_topic").value

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
                self._joint_index = [msg.name.index(j) for j in UR5_JOINT_ORDER]
            except ValueError as e:
                self.get_logger().warn(f"joint state missing UR5 joint: {e}")
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
