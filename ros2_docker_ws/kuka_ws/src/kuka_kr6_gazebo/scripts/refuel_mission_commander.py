#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Import our custom IK-Geo geometric solver
from ik_geometric import IK_spherical_2_parallel

# Kinematics structure exactly as analyzed from the URDF
KIN_KR6_R700 = {
    'H': np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1]
    ]).T,
    'P': np.array([
        [0, 0, 0.208],
        [0.025, -0.0907, -0.192],
        [0.335, 0, -0.0042],
        [0.141, -0.025, -0.0865],
        [0, 0.0505, -0.224],
        [0.0615, 0, -0.0505],
        [0, 0, -0.0285]
    ]).T
}

class RefuelCommander(Node):
    def __init__(self):
        super().__init__('refuel_mission_commander')
        self._action_client = ActionClient(self, FollowJointTrajectory, '/kr6_arm_controller/follow_joint_trajectory')
        
    def euler_to_rot_matrix(self, roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        return R_z @ R_y @ R_x

    def compute_exact_ik(self, target_xyz, target_rpy):
        """Uses IK-Geo to compute exact joint states for a Cartesial goal"""
        self.get_logger().info(f"Computing Exact Inverse Kinematics for target: {target_xyz}")
        
        R_06 = self.euler_to_rot_matrix(*target_rpy)
        p_0T = np.array(target_xyz)[:, None]
        
        # We query the exact geometric solver
        # Returns [6 x N] array of all N valid exact mathematical joints
        solutions = IK_spherical_2_parallel(R_06, p_0T, KIN_KR6_R700)
        
        if solutions.shape[1] == 0:
            self.get_logger().error("No exact IK solutions found!")
            return None
            
        self.get_logger().info(f"Found {solutions.shape[1]} valid algebraic solutions. Selecting the first valid trajectory.")
        # For simplicity, returning the first solution
        return solutions[:, 0].tolist()

    def send_refuel_trajectory(self, target_joints):
        self.get_logger().info('Connecting to Gazebo JointTrajectoryController...')
        self._action_client.wait_for_server()
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start.sec = 3
        
        goal_msg.trajectory.points = [point]

        self.get_logger().info('Moving robot to exact refueling coordinate...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted! Commencing robotic refueling motion.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Motion Complete! Commencing 7 second Refueling Protocol. Wait.')
        
        # Node can trigger a 7 second sleep via sleep() or timers
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    action_client = RefuelCommander()
    
    # Inlet coordinates defined in Gazebo World
    target_xyz = [0.300, 0.400, 0.250]
    target_rpy = [0.0, 0.0, 0.0]  # Assuming forward approach
    
    target_joints = action_client.compute_exact_ik(target_xyz, target_rpy)
    
    if target_joints:
        action_client.send_refuel_trajectory(target_joints)
        rclpy.spin(action_client)


if __name__ == '__main__':
    main()
