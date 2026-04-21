#!/usr/bin/env python3
"""
Admittance Controller — ROS Node for UR5 Force-Compliant Execution
===================================================================

Wraps the standalone AdmittanceController math (M*a + D*v + K*x = F_ext)
in a ROS node that:
  - Subscribes to /ur5/ft_sensor/raw (WrenchStamped) from Gazebo F/T plugin
  - Receives nominal joint waypoints from the mission planner
  - Computes compliant offsets when external forces exceed a threshold
  - Publishes modified joint commands to ur5_arm_controller/command

Three operating modes:
  RIGID:     ||F_ext|| < force_threshold  -> pass nominal trajectory through
  COMPLIANT: force_threshold <= ||F_ext|| < force_abort -> yield via admittance
  ABORT:     ||F_ext|| >= force_abort     -> halt and signal safety abort

Run standalone:  python3 admittance_node.py
Run with mission: launched automatically by refuel_mission.py --compliant
"""
import sys
import os
import threading
import numpy as np

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))

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


# ── Admittance ROS Node ──────────────────────────────────────────

class AdmittanceNode:
    """
    ROS node that adds force-compliant behavior to UR5 trajectory execution.

    Architecture:
      Mission Planner  -->  [nominal waypoints]  -->  AdmittanceNode
      F/T Sensor       -->  [WrenchStamped]      -->  AdmittanceNode
      AdmittanceNode   -->  [modified joints]    -->  ur5_arm_controller
    """

    # Operating mode constants
    MODE_RIGID = 'RIGID'
    MODE_COMPLIANT = 'COMPLIANT'
    MODE_ABORT = 'ABORT'

    def __init__(self):
        import rospy
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import String
        from sensor_msgs.msg import JointState

        rospy.init_node('admittance_controller', anonymous=True)

        # Parameters
        self.mass = rospy.get_param('~mass', 2.0)
        self.damping = rospy.get_param('~damping', 20.0)
        self.stiffness = rospy.get_param('~stiffness', 100.0)
        self.force_threshold = rospy.get_param('~force_threshold', 15.0)
        self.force_abort = rospy.get_param('~force_abort', 50.0)
        self.update_rate = rospy.get_param('~update_rate', 50)

        # Admittance math
        self.controller = AdmittanceController(
            mass=self.mass, damping=self.damping, stiffness=self.stiffness)

        # State
        self.latest_wrench = np.zeros(6)  # [Fx, Fy, Fz, Tx, Ty, Tz]
        self.current_joints = np.zeros(6)
        self.nominal_joints = None  # Set by planner
        self.mode = self.MODE_RIGID
        self.aborted = False
        self._lock = threading.Lock()

        # Subscribers
        self.ft_sub = rospy.Subscriber(
            '/ur5/ft_sensor/raw', WrenchStamped, self._ft_callback)
        self.joint_sub = rospy.Subscriber(
            '/joint_states', JointState, self._joint_callback)

        # Publishers
        self.status_pub = rospy.Publisher(
            '/ur5/admittance/status', String, queue_size=10)
        self._cmd_pub = None  # Lazy-init for trajectory command

        rospy.loginfo(
            f"[Admittance] Node ready: M={self.mass}, D={self.damping}, "
            f"K={self.stiffness}, threshold={self.force_threshold}N, "
            f"abort={self.force_abort}N, rate={self.update_rate}Hz")

    def _ft_callback(self, msg):
        """Store latest force/torque measurement."""
        with self._lock:
            self.latest_wrench = np.array([
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
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
            wrench = self.latest_wrench.copy()
            q_nominal = self.nominal_joints.copy() if self.nominal_joints is not None else None
            q_current = self.current_joints.copy()

        if q_nominal is None:
            return q_current, self.MODE_RIGID

        force_3d = wrench[:3]  # Use translational forces only
        force_magnitude = np.linalg.norm(force_3d)

        # Mode selection
        if force_magnitude >= self.force_abort:
            self.mode = self.MODE_ABORT
            self.aborted = True
            return q_current, self.MODE_ABORT

        if force_magnitude < self.force_threshold:
            self.mode = self.MODE_RIGID
            self.controller.reset()
            return q_nominal, self.MODE_RIGID

        # COMPLIANT mode: compute workspace offset, map to joint space
        self.mode = self.MODE_COMPLIANT
        dt = 1.0 / self.update_rate

        # Admittance law: M*a + D*v + K*x = F_ext  ->  workspace offset
        workspace_offset = self.controller.update(force_3d, dt)

        # Map workspace offset to joint offset via J^T
        # dq = J^T * dx  (Jacobian transpose mapping)
        J = numerical_jacobian(q_current, KIN_UR5)
        J_pos = J[:3, :]  # Position rows only (3x6)
        dq = J_pos.T @ workspace_offset

        # Clamp joint offset to prevent large jumps
        max_dq = 0.1  # rad, ~5.7 degrees max compliance per joint
        dq = np.clip(dq, -max_dq, max_dq)

        q_command = q_nominal + dq
        return q_command, self.MODE_COMPLIANT

    def publish_command(self, q_command):
        """Publish joint command to the UR5 trajectory controller."""
        import rospy
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        if self._cmd_pub is None:
            self._cmd_pub = rospy.Publisher(
                '/ur5_arm_controller/command',
                JointTrajectory, queue_size=10)
            rospy.sleep(0.2)

        msg = JointTrajectory()
        msg.joint_names = UR5_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = q_command.tolist()
        pt.velocities = [0.0] * 6
        pt.time_from_start = rospy.Duration.from_sec(1.0 / self.update_rate)
        msg.points.append(pt)
        self._cmd_pub.publish(msg)

    def publish_status(self):
        """Broadcast current mode on /ur5/admittance/status."""
        from std_msgs.msg import String
        self.status_pub.publish(String(data=self.mode))

    def execute_waypoint(self, q_nominal, duration=0.05):
        """
        Execute a single waypoint with admittance compliance.

        Sets the nominal target and runs the admittance loop for `duration`
        seconds (default: one timestep). Returns the final mode.

        This is the API used by refuel_mission.py send_trajectory_compliant().
        """
        import rospy

        self.set_nominal(q_nominal)
        rate = rospy.Rate(self.update_rate)
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
        import rospy

        rospy.loginfo(f"[Admittance] Executing {len(trajectory)} waypoints "
                      f"(dt={dt}s, compliant)")
        self.controller.reset()
        self.aborted = False

        for i, q_wp in enumerate(trajectory):
            mode = self.execute_waypoint(q_wp, duration=dt)

            if mode == self.MODE_ABORT:
                rospy.logwarn(f"[Admittance] ABORT at waypoint {i}/{len(trajectory)} "
                              f"— force exceeded {self.force_abort}N")
                return False

            if mode == self.MODE_COMPLIANT:
                force_mag = np.linalg.norm(self.latest_wrench[:3])
                rospy.loginfo(f"[Admittance] wp {i}: COMPLIANT "
                              f"(F={force_mag:.1f}N, offset={np.linalg.norm(self.controller.offset):.4f}m)")

        rospy.loginfo("[Admittance] Trajectory complete")
        return True


# ── Standalone test ───────────────────────────────────────────────

def _standalone_test():
    """Run admittance node standalone for testing with rostopic pub."""
    import rospy

    node = AdmittanceNode()
    rate = rospy.Rate(node.update_rate)

    rospy.loginfo("[Admittance] Standalone mode — waiting for F/T data on "
                  "/ur5/ft_sensor/raw and joint states on /joint_states")
    rospy.loginfo("[Admittance] Set nominal joints via: "
                  "rostopic pub /ur5/admittance/target_joints ...")

    # Use current joint state as nominal target
    rospy.sleep(1.0)
    node.set_nominal(node.current_joints.copy())

    while not rospy.is_shutdown():
        q_cmd, mode = node.compute_compliant_joints()
        node.publish_command(q_cmd)
        node.publish_status()

        force_mag = np.linalg.norm(node.latest_wrench[:3])
        if force_mag > 0.1:
            rospy.loginfo_throttle(1.0,
                f"[Admittance] mode={mode}, ||F||={force_mag:.2f}N, "
                f"offset={np.linalg.norm(node.controller.offset):.4f}m")
        rate.sleep()


if __name__ == '__main__':
    _standalone_test()
