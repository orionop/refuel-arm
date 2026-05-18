#!/usr/bin/env python
"""
NEO controller as a ROS1 (Noetic) node for Franka Emika Panda + Gazebo.

Same algorithm and structure as the UR5 variant, but with Panda-specific
defaults: 7 joints, panda_link* link names, panda_joint* joint names, and
the original NEO paper scene (Haviland & Corke 2021, Fig 1).

Subscribes:
    /joint_states  (sensor_msgs/JointState)

Publishes:
    /<controller>/command  (trajectory_msgs/JointTrajectory)
        default controller name: "panda_arm_controller"

Reads obstacles from the Gazebo /gazebo/model_states topic. Any model whose
name starts with the OBSTACLE_PREFIX (default "obs_") is treated as a moving
spherical obstacle.

Assumptions:
    - ROS1 Noetic, Python 3.8+ (rospy)
    - Packages on the lab PC:
        rospy, roboticstoolbox-python, spatialgeometry, spatialmath-python,
        numpy, qpsolvers (with quadprog), gazebo_msgs, trajectory_msgs,
        sensor_msgs, geometry_msgs
    - Panda model loaded via `roboticstoolbox.models.Panda()`
    - franka_description / franka_gazebo (or equivalent) provides the URDF
      and Gazebo plugin on the lab PC
    - Controller takes JointTrajectory commands (default JointTrajectoryController)

Usage:
    rosrun <pkg> neo_ros_node_panda.py
    or:
    python3 neo_ros_node_panda.py
"""

import threading
import numpy as np

import rospy
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from gazebo_msgs.msg import ModelStates

# ===== Parameters (all overridable via rosparam) =====

CONTROLLER_TOPIC_DEFAULT = "/panda_arm_controller/command"
JOINT_NAMES_DEFAULT = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
OBSTACLE_PREFIX_DEFAULT = "obs_"
# Target from the original NEO paper Panda demo
TARGET_DEFAULT = [0.6, -0.2, 0.0]
CONTROL_RATE_DEFAULT = 100.0
INFLUENCE_DIST_DEFAULT = 0.30
STOP_DIST_DEFAULT = 0.05
OBSTACLE_RADIUS_DEFAULT = 0.05


class NEONode:
    def __init__(self):
        rospy.init_node("neo_controller", anonymous=False)

        # Params
        self.controller_topic = rospy.get_param(
            "~controller_topic", CONTROLLER_TOPIC_DEFAULT
        )
        self.joint_names = rospy.get_param("~joint_names", JOINT_NAMES_DEFAULT)
        self.obstacle_prefix = rospy.get_param(
            "~obstacle_prefix", OBSTACLE_PREFIX_DEFAULT
        )
        target_xyz = rospy.get_param("~target", TARGET_DEFAULT)
        self.control_rate = rospy.get_param("~control_rate", CONTROL_RATE_DEFAULT)
        self.di = rospy.get_param("~influence_dist", INFLUENCE_DIST_DEFAULT)
        self.ds = rospy.get_param("~stop_dist", STOP_DIST_DEFAULT)
        self.obs_radius = rospy.get_param("~obstacle_radius", OBSTACLE_RADIUS_DEFAULT)

        # Robot model
        self.robot = rtb.models.Panda()
        self.n = 7

        # Initial guess uses Panda's ready pose (qr); replaced when /joint_states arrives
        self.q = self.robot.qr.copy()
        self.have_joint_state = False
        self.joint_lock = threading.Lock()

        # Obstacles
        self.obstacles = {}
        self.obs_lock = threading.Lock()

        # Target pose: keep orientation from initial fkine, set xyz from param
        self.Tep = self.robot.fkine(self.q)
        self.Tep.A[:3, 3] = np.array(target_xyz)

        # Pub/sub
        self.cmd_pub = rospy.Publisher(
            self.controller_topic, JointTrajectory, queue_size=10
        )
        rospy.Subscriber("/joint_states", JointState, self.joint_state_cb, queue_size=10)
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_cb, queue_size=10
        )

        rospy.loginfo(
            "[neo_controller] Panda variant init complete. publishing to %s",
            self.controller_topic,
        )

    # ---- Callbacks ----

    def joint_state_cb(self, msg: JointState):
        """Update self.q from incoming joint states (reorder to expected order)."""
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            q_new = np.array([name_to_pos[n] for n in self.joint_names])
        except KeyError:
            return
        with self.joint_lock:
            self.q = q_new
            self.have_joint_state = True

    def model_states_cb(self, msg: ModelStates):
        """Pick up obstacles by name prefix; record pose and velocity."""
        with self.obs_lock:
            for i, name in enumerate(msg.name):
                if not name.startswith(self.obstacle_prefix):
                    continue
                p = msg.pose[i].position
                v = msg.twist[i].linear
                pose = sm.SE3(p.x, p.y, p.z)

                if name not in self.obstacles:
                    sph = sg.Sphere(radius=self.obs_radius, pose=pose)
                else:
                    sph = self.obstacles[name]
                    sph.T = pose.A
                sph.v = [v.x, v.y, v.z, 0.0, 0.0, 0.0]
                self.obstacles[name] = sph

    # ---- NEO QP step ----

    def neo_step(self):
        """Solve one NEO QP step. Returns (qd, arrived) or None if infeasible."""
        with self.joint_lock:
            if not self.have_joint_state:
                return None
            q = self.q.copy()

        robot = self.robot
        n = self.n
        robot.q = q

        Te = robot.fkine(q)
        eTep = Te.inv() * self.Tep
        e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

        v, arrived = rtb.p_servo(Te, self.Tep, 0.5, 0.01)

        Y = 0.01
        Q = np.eye(n + 6)
        Q[:n, :n] *= Y
        slack_gain = min(1.0 / max(e, 1e-3), 1e3)
        Q[n:, n:] = slack_gain * np.eye(6)

        Aeq = np.c_[robot.jacobe(q), np.eye(6)]
        beq = v.reshape((6,))

        Ain = np.zeros((n + 6, n + 6))
        bin_ = np.zeros(n + 6)
        ps = 0.05
        pi_inf = 0.9
        Ain[:n, :n], bin_[:n] = robot.joint_velocity_damper(ps, pi_inf, n)

        with self.obs_lock:
            obstacles_snapshot = list(self.obstacles.values())

        for collision in obstacles_snapshot:
            try:
                c_Ain, c_bin = robot.link_collision_damper(
                    collision,
                    q[:n],
                    self.di,
                    self.ds,
                    1.0,
                    start=robot.link_dict["panda_link1"],
                    end=robot.link_dict["panda_hand"],
                )
            except Exception:
                continue
            if c_Ain is not None and c_bin is not None:
                c_Ain = c_Ain[:, :n]
                c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]
                Ain = np.r_[Ain, c_Ain]
                bin_ = np.r_[bin_, c_bin]

        c = np.r_[-robot.jacobm(q).reshape((n,)), np.zeros(6)]

        qdlim = robot.qdlim[:n] if hasattr(robot, "qdlim") else np.pi * np.ones(n)
        lb = -np.r_[qdlim, 10 * np.ones(6)]
        ub = np.r_[qdlim, 10 * np.ones(6)]

        try:
            qd_full = qp.solve_qp(
                Q, c, Ain, bin_, Aeq, beq, lb=lb, ub=ub, solver="quadprog"
            )
        except Exception as exc:
            rospy.logwarn("[neo_controller] QP error: %s", exc)
            return None

        if qd_full is None:
            return None
        return qd_full[:n], arrived

    # ---- Main control loop ----

    def run(self):
        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate
        rospy.loginfo("[neo_controller] waiting for first /joint_states...")
        while not rospy.is_shutdown() and not self.have_joint_state:
            rate.sleep()
        rospy.loginfo("[neo_controller] got first joint state, starting NEO loop")

        infeasible_streak = 0
        while not rospy.is_shutdown():
            result = self.neo_step()
            if result is None:
                infeasible_streak += 1
                if infeasible_streak % 50 == 1:
                    rospy.logwarn(
                        "[neo_controller] QP infeasible (streak=%d), holding pose",
                        infeasible_streak,
                    )
                rate.sleep()
                continue

            qd, arrived = result
            infeasible_streak = 0

            with self.joint_lock:
                q_now = self.q.copy()

            # Use a longer trajectory horizon than the publish period so the
            # JointTrajectoryController PID has time to actually track the goal.
            # Integrating qd over `horizon` and giving the controller `horizon`
            # seconds to reach it (re-targeted every 1/control_rate seconds).
            horizon = max(5.0 * dt, 0.05)
            q_next = q_now + qd * horizon

            msg = JointTrajectory()
            msg.header.stamp = rospy.Time.now()
            msg.joint_names = list(self.joint_names)
            pt = JointTrajectoryPoint()
            pt.positions = q_next.tolist()
            pt.velocities = qd.tolist()
            pt.time_from_start = rospy.Duration.from_sec(horizon)
            msg.points.append(pt)
            self.cmd_pub.publish(msg)

            if arrived:
                rospy.loginfo_throttle(1.0, "[neo_controller] target reached")

            rate.sleep()


def main():
    node = NEONode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
