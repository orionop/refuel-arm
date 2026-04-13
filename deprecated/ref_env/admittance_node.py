#!/usr/bin/env python3
"""
Admittance Controller — ROS2 Node for UR5 Force-Compliant Execution
====================================================================

Wraps the standalone AdmittanceController math (M*a + D*v + K*x = F_ext)
in a ROS2 rclpy node that:
  - Subscribes to /ur5/ft_sensor/raw (WrenchStamped) from Gazebo F/T plugin
  - Receives nominal joint waypoints from the mission planner
  - Computes compliant offsets when external forces exceed a threshold
  - Publishes modified joint commands to ur5_arm_controller/command

Three operating modes:
  RIGID:     ||F_ext|| < force_threshold  -> pass nominal trajectory through
  COMPLIANT: force_threshold <= ||F_ext|| < force_abort -> yield via admittance
  ABORT:     ||F_ext|| >= force_abort     -> halt and signal safety abort

ROS1 equivalent: deprecated/admittance_node.ros1.py

Run standalone:  python3 admittance_node.py
Run with mission: launched automatically by refuel_mission.py --compliant
"""
import sys
import os
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RosDuration

from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as BuiltinDuration

# ── Path setup (local ref_env imports) ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from admittance_controller import AdmittanceController
from ik_geometric import fwd_kinematics, KIN_UR5


# ── Numerical Jacobian ────────────────────────────────────────────

def numerical_jacobian(q, kin=None, eps=1e-6):
    """
    Compute 6x6 numerical Jacobian (position rows 0-2, orientation rows 3-5).
    """
    if kin is None:
        kin = KIN_UR5
    J = np.zeros((6, 6))
    R0, p0 = fwd_kinematics(q, kin)

    for j in range(6):
        q_plus = q.copy()
        q_plus[j] += eps
        R_plus, p_plus = fwd_kinematics(q_plus, kin)

        # Positional Jacobian (rows 0-2)
        J[:3, j] = (p_plus - p0) / eps

        # Rotational Jacobian (rows 3-5) via R_delta -> axis-angle
        R_delta = R_plus @ R0.T
        J[3, j] = (R_delta[2, 1] - R_delta[1, 2]) / (2 * eps)
        J[4, j] = (R_delta[0, 2] - R_delta[2, 0]) / (2 * eps)
        J[5, j] = (R_delta[1, 0] - R_delta[0, 1]) / (2 * eps)

    return J


# ── UR5 Joint Names ──────────────────────────────────────────────

UR5_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]


# ── Admittance ROS2 Node ─────────────────────────────────────────

class AdmittanceNode(Node):
    """
    ROS2 node that adds force-compliant behavior to UR5 trajectory execution.

    Architecture:
      Mission Planner  -->  [nominal waypoints]  -->  AdmittanceNode
      F/T Sensor       -->  [WrenchStamped]      -->  AdmittanceNode
      AdmittanceNode   -->  [modified joints]    -->  ur5_arm_controller

    ROS1→ROS2 changes (vs deprecated/admittance_node.ros1.py):
      rospy.init_node()        → super().__init__()
      rospy.get_param()        → self.declare_parameter() + get_parameter()
      rospy.Subscriber()       → self.create_subscription()
      rospy.Publisher()        → self.create_publisher()
      rospy.Rate()             → self.create_rate()
      rospy.is_shutdown()      → rclpy.ok()
      rospy.sleep()            → time.sleep()
      rospy.Duration.from_sec()→ BuiltinDuration(sec=..., nanosec=...)
      rospy.loginfo/warn()     → self.get_logger().info/warn()
    """

    MODE_RIGID    = 'RIGID'
    MODE_COMPLIANT = 'COMPLIANT'
    MODE_ABORT    = 'ABORT'

    def __init__(self):
        super().__init__('admittance_controller')

        # Parameters (ROS2: declare first, then read)
        self.declare_parameter('mass',                2.0)
        self.declare_parameter('damping',            20.0)
        self.declare_parameter('stiffness',         100.0)
        self.declare_parameter('force_threshold',    15.0)
        self.declare_parameter('force_abort',        50.0)
        self.declare_parameter('update_rate',        50)

        self.mass            = self.get_parameter('mass').value
        self.damping         = self.get_parameter('damping').value
        self.stiffness       = self.get_parameter('stiffness').value
        self.force_threshold = self.get_parameter('force_threshold').value
        self.force_abort     = self.get_parameter('force_abort').value
        self.update_rate     = self.get_parameter('update_rate').value

        # Admittance math
        self.controller = AdmittanceController(
            mass=self.mass, damping=self.damping, stiffness=self.stiffness)

        # State
        self.latest_wrench  = np.zeros(6)
        self.current_joints = np.zeros(6)
        self.nominal_joints = None
        self.mode           = self.MODE_RIGID
        self.aborted        = False
        self._lock          = threading.Lock()

        # Subscribers
        self.ft_sub = self.create_subscription(
            WrenchStamped, '/ur5/ft_sensor/raw', self._ft_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/ur5/admittance/status', 10)
        self._cmd_pub   = None  # Lazy-init for trajectory command

        self.get_logger().info(
            f"[Admittance] Node ready: M={self.mass}, D={self.damping}, "
            f"K={self.stiffness}, threshold={self.force_threshold}N, "
            f"abort={self.force_abort}N, rate={self.update_rate}Hz")

    # ── Callbacks ────────────────────────────────────────────────

    def _ft_callback(self, msg):
        """Store latest force/torque measurement."""
        with self._lock:
            self.latest_wrench = np.array([
                msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ])

    def _joint_callback(self, msg):
        """Store latest joint state (ordered by UR5 joint names)."""
        try:
            indices = [msg.name.index(n) for n in UR5_JOINT_NAMES]
            with self._lock:
                self.current_joints = np.array([msg.position[i] for i in indices])
        except (ValueError, IndexError):
            pass

    # ── Control logic ────────────────────────────────────────────

    def set_nominal(self, q_nominal):
        """Set the nominal (planned) joint target for the current waypoint."""
        with self._lock:
            self.nominal_joints = np.array(q_nominal, dtype=float)

    def compute_compliant_joints(self):
        """
        Core admittance computation: nominal joints + force-compliant offset.

        Returns (q_command, mode_string).
        """
        with self._lock:
            wrench    = self.latest_wrench.copy()
            q_nominal = self.nominal_joints.copy() if self.nominal_joints is not None else None
            q_current = self.current_joints.copy()

        if q_nominal is None:
            return q_current, self.MODE_RIGID

        force_3d        = wrench[:3]
        force_magnitude = np.linalg.norm(force_3d)

        if force_magnitude >= self.force_abort:
            self.mode    = self.MODE_ABORT
            self.aborted = True
            return q_current, self.MODE_ABORT

        if force_magnitude < self.force_threshold:
            self.mode = self.MODE_RIGID
            self.controller.reset()
            return q_nominal, self.MODE_RIGID

        # COMPLIANT mode
        self.mode = self.MODE_COMPLIANT
        dt = 1.0 / self.update_rate

        workspace_offset = self.controller.update(force_3d, dt)

        J     = numerical_jacobian(q_current, KIN_UR5)
        J_pos = J[:3, :]
        dq    = J_pos.T @ workspace_offset
        dq    = np.clip(dq, -0.1, 0.1)

        return q_nominal + dq, self.MODE_COMPLIANT

    def publish_command(self, q_command):
        """Publish joint command to the UR5 trajectory controller."""
        if self._cmd_pub is None:
            self._cmd_pub = self.create_publisher(
                JointTrajectory, '/ur5_arm_controller/command', 10)
            time.sleep(0.2)

        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions  = q_command.tolist()
        pt.velocities = [0.0] * 6

        # ROS2: time_from_start uses builtin_interfaces/Duration
        dt = 1.0 / self.update_rate
        pt.time_from_start = BuiltinDuration(
            sec=int(dt),
            nanosec=int((dt % 1) * 1_000_000_000))

        msg.points.append(pt)
        self._cmd_pub.publish(msg)

    def publish_status(self):
        """Broadcast current mode on /ur5/admittance/status."""
        self.status_pub.publish(String(data=self.mode))

    # ── Execution API (used by refuel_mission.py) ─────────────────

    def execute_waypoint(self, q_nominal, duration=0.05):
        """
        Execute a single waypoint with admittance compliance.

        Runs the admittance loop for `duration` seconds. Returns final mode.
        """
        self.set_nominal(q_nominal)
        rate    = self.create_rate(self.update_rate)
        n_steps = max(1, int(duration * self.update_rate))

        for _ in range(n_steps):
            if self.aborted:
                self.publish_status()
                return self.MODE_ABORT

            q_cmd, mode = self.compute_compliant_joints()
            self.publish_command(q_cmd)
            self.publish_status()
            rate.sleep()

        return self.mode

    def execute_trajectory(self, trajectory, dt=0.05):
        """
        Execute a full trajectory with admittance compliance.

        Parameters
        ----------
        trajectory : (N, 6) array — joint waypoints
        dt : float — time per waypoint (seconds)

        Returns
        -------
        success : bool — True if completed, False if aborted
        """
        self.get_logger().info(
            f"[Admittance] Executing {len(trajectory)} waypoints "
            f"(dt={dt}s, compliant)")
        self.controller.reset()
        self.aborted = False

        for i, q_wp in enumerate(trajectory):
            mode = self.execute_waypoint(q_wp, duration=dt)

            if mode == self.MODE_ABORT:
                self.get_logger().warn(
                    f"[Admittance] ABORT at waypoint {i}/{len(trajectory)} "
                    f"— force exceeded {self.force_abort}N")
                return False

            if mode == self.MODE_COMPLIANT:
                force_mag = np.linalg.norm(self.latest_wrench[:3])
                self.get_logger().info(
                    f"[Admittance] wp {i}: COMPLIANT "
                    f"(F={force_mag:.1f}N, offset={np.linalg.norm(self.controller.offset):.4f}m)")

        self.get_logger().info("[Admittance] Trajectory complete")
        return True


# ── Standalone test ───────────────────────────────────────────────

def _standalone_test():
    """Run admittance node standalone for testing with ros2 topic pub."""
    rclpy.init()
    node = AdmittanceNode()

    node.get_logger().info(
        "[Admittance] Standalone mode — waiting for F/T data on "
        "/ur5/ft_sensor/raw and joint states on /joint_states")

    time.sleep(1.0)
    with node._lock:
        node.set_nominal(node.current_joints.copy())

    _last_log = [0.0]

    def spin_once_loop():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            q_cmd, mode = node.compute_compliant_joints()
            node.publish_command(q_cmd)
            node.publish_status()

            force_mag = np.linalg.norm(node.latest_wrench[:3])
            now = time.monotonic()
            if force_mag > 0.1 and (now - _last_log[0]) >= 1.0:
                node.get_logger().info(
                    f"[Admittance] mode={mode}, ||F||={force_mag:.2f}N, "
                    f"offset={np.linalg.norm(node.controller.offset):.4f}m")
                _last_log[0] = now

            time.sleep(1.0 / node.update_rate)

    try:
        spin_once_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    _standalone_test()
