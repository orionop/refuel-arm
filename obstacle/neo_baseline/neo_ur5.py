#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import cProfile
import time

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a UR5 robot object
panda = rtb.models.UR5()

# Set joint angles to a standing-up "ready" configuration.
# UR5 joint order: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# Matches ROS1 launch file: shoulder_lift at -pi/2 to stand the arm upright.
panda.q = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])

# Number of joints in the UR5 which we are controlling
n = 6

# Make two obstacles with velocities.
# Start far in +y, move in -y through the workspace.
# Raised z so they pass through the EE region, not the body.
s0 = sg.Sphere(radius=0.05, pose=sm.SE3(0.35, 0.5, 0.5))
s0.v = [0, -0.15, 0, 0, 0, 0]

s1 = sg.Sphere(radius=0.05, pose=sm.SE3(0.1, 0.5, 0.7))
s1.v = [0, -0.15, 0, 0, 0, 0]

collisions = [s0, s1]

# Make a target (within UR5 reach, comfortably reachable)
target = sg.Sphere(radius=0.02, pose=sm.SE3(0.4, -0.4, 0.4))

# Add the Panda and shapes to the simulator
env.add(panda)
env.add(s0)
env.add(s1)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = panda.fkine(panda.q)
Tep.A[:3, 3] = target.T[:3, 3]
# Tep.A[2, 3] += 0.1


def step():
    # The pose of the Panda's end-effector
    Te = panda.fkine(panda.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.01)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    # For each collision in the scene
    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = panda.link_collision_damper(
            collision,
            panda.q[:n],
            0.3,
            0.05,
            1.0,
            start=panda.link_dict["shoulder_link"],
            end=panda.link_dict["tool0"],
        )

        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None:
            c_Ain = c_Ain[:, :n]  # API compat: trim to n joint cols
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm(panda.q).reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    # UR5 spec: ±pi rad/s (±180 deg/s) per joint
    qdlim = np.pi * np.ones(n)
    lb = -np.r_[qdlim, 10 * np.ones(6)]
    ub = np.r_[qdlim, 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver="quadprog")

    # Handle infeasible QP gracefully (keep arm still)
    if qd is None:
        print(f"[WARN] QP infeasible, holding pose. err={e:.4f}")
        panda.qd[:n] = np.zeros(n)
    else:
        # Apply the joint velocities to the robot
        panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.01)
    time.sleep(0.03)  # slow visual playback (no algorithm change)

    return arrived

arrived = False
while not arrived:
    arrived = step()

env.hold()
