#!/usr/bin/env python3
"""
KUKA KR6 R700 — Integrated STOMP Refueling Mission (DIRECT SEQUENCE + VISUALIZATION)
====================================================================================

Mission sequence (DIRECT):
  HOME → YELLOW (pick nozzle) → RED (refuel, avoid BLUE DOT) → YELLOW (drop nozzle) → HOME

Features:
  1. Direct Flight: No return to HOME between YELLOW and RED.
  2. Blue Dot Visualization: ROS Marker in RViz for the obstacle.
  3. Tuned STOMP: Higher avoidance weight and centered obstacle.
  4. Anti-Jitter: Gaussian smoothing.
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
Q_NOZZLE = np.array([0.785, -0.94, 0.94, 0.0, 0.0, 0.0])     # YELLOW
REFUEL_TARGET_XYZ = np.array([0.55, 0.3, 0.5])              # RED
DWELL_TIME = 5.0

# Fixed Orientation for RED
R_RED = ik.rot(np.array([0.0, 1.0, 0.0]), np.radians(15.0))

def solve_goal_ik(pos, R, q_prev):
    Q = ik.IK_spherical_2_parallel(R, pos)
    if Q.size == 0: return None
    best_q, min_dist = None, float('inf')
    for i in range(Q.shape[1]):
        q = (Q[:, i] + np.pi) % (2 * np.pi) - np.pi
        valid = True
        for j in range(6):
            if q[j] < JOINT_LIMITS[j, 0] or q[j] > JOINT_LIMITS[j, 1]: valid = False; break
        if valid:
            dist = np.linalg.norm(q - q_prev)
            if dist < min_dist: min_dist = dist; best_q = q
    return best_q

# --- OBSTACLE CONFIG ---
OBSTACLE_CENTER = np.array([0.52, 0.40, 0.60]) # Right between Yellow and Red

def create_blue_dot_obstacle():
    pc = []
    # Obstacle is a slightly larger "box" of points to ensure the arm sees it
    for x in np.linspace(OBSTACLE_CENTER[0]-0.08, OBSTACLE_CENTER[0]+0.08, 12):
        for y in np.linspace(OBSTACLE_CENTER[1]-0.08, OBSTACLE_CENTER[1]+0.08, 12):
            pc.append([x, y, OBSTACLE_CENTER[2]]) # Elevation wall at 0.6m
    return np.array(pc)

def publish_obstacle_marker():
    """Publish a blue sphere to RViz representing the obstacle."""
    import rospy
    from visualization_msgs.msg import Marker
    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(0.5)
    
    m = Marker()
    m.header.frame_id = "world"; m.header.stamp = rospy.Time.now(); m.ns = "obstacle"; m.id = 99
    m.type = Marker.SPHERE; m.action = Marker.ADD
    m.pose.position.x = OBSTACLE_CENTER[0]; m.pose.position.y = OBSTACLE_CENTER[1]; m.pose.position.z = OBSTACLE_CENTER[2]
    m.pose.orientation.w = 1.0
    m.scale.x = 0.15; m.scale.y = 0.15; m.scale.z = 0.15 # 15cm sphere
    m.color.r = 0.0; m.color.g = 0.3; m.color.b = 1.0; m.color.a = 0.8
    pub.publish(m)
    print(f"🔵 Obstacle Marker published at {OBSTACLE_CENTER}")

def plan_and_smooth(q_start, q_goal, grid=None, name="Segment", n_wp=60):
    print(f"\n[STOMP] Planning: {name} (Grid={bool(grid)})")
    raw_traj, _ = stomp_optimize(
        q_start=q_start, q_goal=q_goal,
        joint_limits=JOINT_LIMITS,
        grid=grid,
        n_waypoints=n_wp,
        n_iterations=150, # More iterations for harder avoidance
        n_rollouts=15,
        w_smooth=80.0,
        w_obs=3000.0,     # Doubled weight for "scary" obstacle
        safety_margin=0.3,
        verbose=False
    )
    smooth_traj = gaussian_filter1d(raw_traj, sigma=1.0, axis=0)
    smooth_traj[0] = q_start; smooth_traj[-1] = q_goal
    return smooth_traj

def execute_ros(trajectory, dt=0.15):
    import os
    ros_python = '/opt/ros/noetic/lib/python3/dist-packages'
    if ros_python not in sys.path and os.path.isdir(ros_python): sys.path.insert(0, ros_python)
    import rospy
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint

    client = actionlib.SimpleActionClient('/kr6_arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    if not client.wait_for_server(timeout=rospy.Duration(5.0)): return False
    
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
    for i, q in enumerate(trajectory):
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist(); pt.velocities = [0.0]*6; pt.time_from_start = rospy.Duration.from_sec(i * dt)
        goal.trajectory.points.append(pt)
    client.send_goal(goal); client.wait_for_result()
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros", action="store_true")
    args = parser.parse_args()

    print("="*65)
    print("🚀 REFINED STOMP MISSION: Yellow <-> Red (Direct Sequence)")
    print("="*65)

    q_red = solve_goal_ik(REFUEL_TARGET_XYZ, R_RED, Q_HOME)
    if q_red is None: return

    blue_dot_pc = create_blue_dot_obstacle()
    grid = Grid3D(resolution=0.04)
    grid.build_from_point_cloud(blue_dot_pc)

    # 3. DIRECT PLANNING SEQUENCE
    # Seg 1: HOME -> YELLOW (Transit)
    seg1 = plan_and_smooth(Q_HOME, Q_NOZZLE, name="HOME -> YELLOW (Pick)")
    # Seg 2: YELLOW -> RED (Avoidance!)
    seg2 = plan_and_smooth(Q_NOZZLE, q_red, grid=grid, name="YELLOW -> RED (Dodging Blue Dot)")
    # Seg 3: RED -> YELLOW (Avoidance!)
    seg3 = plan_and_smooth(q_red, Q_NOZZLE, grid=grid, name="RED -> YELLOW (Return)")
    # Seg 4: YELLOW -> HOME (Finish)
    seg4 = plan_and_smooth(Q_NOZZLE, Q_HOME, name="YELLOW -> HOME (Final)")

    if args.ros:
        import rospy
        rospy.init_node('full_stomp_mission_refined')
        publish_obstacle_marker()
        
        segments = [
            ("Pick Nozzle", seg1, 0.15),
            ("🔴 Approach Refuel (Avoidance)", seg2, 0.2), 
            ("Return Nozzle (Avoidance)", seg3, 0.2),
            ("Park Home", seg4, 0.15)
        ]
        
        for name, traj, dt in segments:
            print(f"  ➜ {name}")
            execute_ros(traj, dt=dt)
            if "Approach" in name: 
                print(f"     ⏱️  Refueling for {DWELL_TIME}s..."); time.sleep(DWELL_TIME)
        
        print("\n🎉 REFINED MISSION COMPLETE")

if __name__ == "__main__":
    main()
