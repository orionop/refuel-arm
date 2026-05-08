#!/usr/bin/env python3
"""Spawn a moving cylinder obstacle in gz sim and publish its pose.

Two responsibilities:

1. **Spawn** a cylinder model in the running gz sim world via the
   `/world/<world>/create` service.
2. **Animate** it along a configurable straight-line trajectory by sending
   pose commands to `/world/<world>/set_pose`, while publishing the same
   pose to `/obstacle_pose` (PoseStamped) so the safety_node can consume
   ground-truth obstacle state.

This intentionally bypasses the camera/perception layer for now — once a
camera + YOLO pipeline is added, that publisher takes over `/obstacle_pose`
and this script becomes the ground-truth reference for evaluation.

Usage (Ubuntu 24 + ROS2 Jazzy + gz sim Harmonic):

    ros2 run obstacle obstacle_simulator \\
        --ros-args -p world:=ur_simulation_gz \\
                   -p start_xyz:=[1.5, 0.0, 0.5] \\
                   -p end_xyz:=[-0.5, 0.0, 0.5] \\
                   -p duration:=3.0 \\
                   -p radius:=0.10
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
except ImportError:                                          # pragma: no cover
    raise SystemExit("[obstacle_simulator] rclpy not available — run on ROS2 Jazzy box.")


_CYLINDER_SDF_TEMPLATE = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia><ixx>0.05</ixx><iyy>0.05</iyy><izz>0.05</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <visual name="vis">
        <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
        <material>
          <ambient>0.9 0.2 0.2 1</ambient>
          <diffuse>0.9 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <collision name="col">
        <geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
      </collision>
    </link>
  </model>
</sdf>
"""


def _gz(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(["gz", *args], capture_output=True, text=True, timeout=timeout)


def spawn_cylinder(world: str, name: str, x: float, y: float, z: float,
                   radius: float, height: float) -> bool:
    sdf = _CYLINDER_SDF_TEMPLATE.format(name=name, x=x, y=y, z=z, r=radius, h=height)
    path = Path(tempfile.gettempdir()) / f"{name}.sdf"
    path.write_text(sdf)

    # remove a stale model with the same name (silent fail on first run)
    _gz("service", "-s", f"/world/{world}/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", f'name: "{name}" type: 2')

    res = _gz("service", "-s", f"/world/{world}/create",
              "--reqtype", "gz.msgs.EntityFactory",
              "--reptype", "gz.msgs.Boolean",
              "--timeout", "5000",
              "--req", f'sdf_filename: "{path}" name: "{name}"')
    return res.returncode == 0


def set_pose(world: str, name: str, p: np.ndarray) -> None:
    req = (f'name: "{name}", position: {{x: {float(p[0])}, '
           f'y: {float(p[1])}, z: {float(p[2])}}}')
    _gz("service", "-s", f"/world/{world}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "1000",
        "--req", req)


class ObstacleSimulator(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_simulator")
        self.declare_parameter("world", "ur_simulation_gz")
        self.declare_parameter("name", "moving_obstacle")
        self.declare_parameter("start_xyz", [1.5, 0.0, 0.5])
        self.declare_parameter("end_xyz", [-0.5, 0.0, 0.5])
        self.declare_parameter("duration", 3.0)
        self.declare_parameter("radius", 0.10)
        self.declare_parameter("height", 1.0)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("loop", False)

        self.world = self.get_parameter("world").value
        self.name = self.get_parameter("name").value
        self.start = np.asarray(self.get_parameter("start_xyz").value, dtype=float)
        self.end = np.asarray(self.get_parameter("end_xyz").value, dtype=float)
        self.duration = float(self.get_parameter("duration").value)
        self.radius = float(self.get_parameter("radius").value)
        self.height = float(self.get_parameter("height").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.loop_traj = bool(self.get_parameter("loop").value)

        self.pose_pub = self.create_publisher(PoseStamped, "/obstacle_pose", 10)

        ok = spawn_cylinder(self.world, self.name,
                            *self.start, self.radius, self.height)
        if not ok:
            self.get_logger().warn(
                "spawn service did not return success — model may already exist")

        self._t0 = self.get_clock().now()
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"obstacle_simulator: {self.name} {self.start} -> {self.end} "
            f"in {self.duration}s, world={self.world}")

    def _tick(self) -> None:
        elapsed = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        if self.loop_traj:
            elapsed = elapsed % self.duration
        else:
            elapsed = min(elapsed, self.duration)
        alpha = elapsed / self.duration
        p = (1.0 - alpha) * self.start + alpha * self.end

        set_pose(self.world, self.name, p)

        msg = PoseStamped()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleSimulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
