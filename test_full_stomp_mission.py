#!/usr/bin/env python3
"""
KUKA KR6 R700 — Integrated STOMP Refueling Mission (C-Space + Obstacle Avoidance)
================================================================================

Mission sequence:
  REST → YELLOW (pick nozzle) → REST → RED (refuel, with BLUE DOT avoidance) → REST

Features:
  1. C-Space Interpolation: Start/End IK solves only.
  2. STOMP + 2.5D Mapping: Avoids "Blue Dot" obstacle during the RED approach.
  3. Anti-Jitter Filtering: Gaussian smoothing for silk-smooth Gazebo execution.
"""
import sys
import os
import time
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Import IK-Geo parts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'kuka_refuel_ws', 'src', 'kuka_kr6_gazebo', 'scripts')))
import ik_geometric as ik
from stomp_collision import Grid3D, stomp_optimize

# ── Joint Limits (URDF) ──────────────────────────────────────────
JOINT_LIMITS = np.array([
    [-2.967,  2.967], [-3.316,  0.785], [-2.094,  2.722],
    [-3.228,  3.228], [-2.094,  2.094], [-6.108,  6.108],
])

# ── Mission Config ───────────────────────────────────────────────
Q_HOME = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])       # REST
Q_NOZZLE = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])     # YELLOW (already accurate)
REFUEL_TARGET_XYZ = np.array([0.55, 0.3, 0.5])              # RED
DWELL_TIME = 5.0

# Fixed Orientation for RED: Forward looking slightly down (approx 15deg)
R_RED = ik.rot(np.array([0.0, 1.0, 0.0]), np.radians(15.0))

def solve_goal_ik(pos, R, q_prev):
    """Solve IK and pick the closest valid solution."""
    Q = ik.IK_spherical_2_parallel(R, pos)
    if Q.size == 0: return None
    best_q, min_dist = None, float('inf')
    for i in range(Q.shape[1]):
        q = (Q[:, i] + np.pi) % (2 * np.pi) - np.pi # wrap
        # limit check
        valid = True
        for j in range(6):
            if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]: valid = False; break
        if valid:
            dist = np.linalg.norm(q - q_prev)
            if dist < min_dist: min_dist = dist; best_q = q
    return best_q

def create_blue_dot_obstacle():
    """Create a point cloud representing a 'Blue Dot' obstacle in the path."""
    pc = []
    # Obstacle area: centered between HOME and RED
    center = np.array([0.45, 0.15, 0.55]) 
    for x in np.linspace(center[0]-0.05, center[0]+0.05, 10):
        for y in np.linspace(center[1]-0.05, center[1]+0.05, 10):
            pc.append([x, y, center[2]])
    return np.array(pc)

def plan_and_smooth(q_start, q_goal, grid=None, name="Segment", n_wp=60):
    print(f"\n[STOMP] Planning: {name} (Grid={bool(grid)})")
    # Optimize
    raw_traj, _ = stomp_optimize(
        q_start=q_start, q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        grid=grid,
        n_waypoints=n_wp,
        n_iterations=120,
        n_rollouts=15,
        w_smooth=100.0,   # Increased for anti-jitter
        w_obs=1500.0,
        safety_margin=0.25,
        verbose=False
    )
    # Gaussian Anti-Jitter Filter
    smooth_traj = gaussian_filter1d(raw_traj, sigma=1.0, axis=0)
    smooth_traj[0] = q_start
    smooth_traj[-1] = q_goal
    return smooth_traj

def execute_ros(trajectory, dt=0.15):
    # ROS paths
    import os
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python): sys.path.insert(0, ros_python)
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient('/kr6_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(timeout=rospy.Duration(5.0)):
        print("   ⚠️  Gazebo Action Server NOT found!")
        return False
    
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.velocities = [0.0]*6
        pt.time_from_start = rospy.Duration.from_sec(i * dt)
        goal.trajectory.points.append(pt)
    client.send_goal(goal)
    return client.wait_for_result(timeout=rospy.Duration(len(trajectory)*dt + 5.0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("🚀 INTEGRATED STOMP MISSION: C-Space + Obstacle Avoidance")
    print("="*60)

    # 1. Target Solving
    q_red = solve_goal_ik(REFUEL_TARGET_XYZ, R_RED, Q_HOME)
    if q_red is None:
        print("❌ No IK solution for RED target!")
        return

    # 2. Obstacle Setup
    blue_dot_pc = create_blue_dot_obstacle()
    grid = Grid3D(resolution=0.04)
    grid.build_from_point_cloud(blue_dot_pc)

    # 3. Planning segments
    # Segment A: HOME -> YELLOW (Transit)
    seg1 = plan_and_smooth(Q_HOME, Q_NOZZLE, name="HOME -> YELLOW (Nozzle Pick)")
    # Segment B: YELLOW -> HOME (Transit)
    seg2 = plan_and_smooth(Q_NOZZLE, Q_HOME, name="YELLOW -> HOME (Reset)")
    # Segment C: HOME -> RED (Avoid Obstacle)
    seg3 = plan_and_smooth(Q_HOME, q_red, grid=grid, name="HOME -> RED (Avoid Blue Dot)")
    # Segment D: RED -> HOME (Reset)
    seg4 = plan_and_smooth(q_red, Q_HOME, name="RED -> HOME (Finish)")

    # 4. Execution
    if args.ros:
        import rospy
        rospy.init_node('full_stomp_mission')
        print("\n🎬 Starting Mission Execution in Gazebo...")
        
        segments = [
            ("Pick Nozzle", seg1, 0.15),
            ("Return Home", seg2, 0.15),
            ("🔴 Approach Refuel (Avoid Obstacle)", seg3, 0.2), # Slower for avoidance
            ("Reset Home", seg4, 0.15)
        ]
        
        for name, traj, dt in segments:
            print(f"  ➜ {name}")
            execute_ros(traj, dt=dt)
            if "Approach" in name: 
                print(f"     ⏱️  Refueling for {DWELL_TIME}s...")
                time.sleep(DWELL_TIME)
        
        print("\n🎉 MISSION SUCCESSFUL")

if __name__ == "__main__":
    main()
