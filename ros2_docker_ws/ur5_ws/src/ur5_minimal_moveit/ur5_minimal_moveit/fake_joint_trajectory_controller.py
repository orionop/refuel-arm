"""
Fake FollowJointTrajectory action server for MoveIt execution without hardware.
Publishes joint_states so RViz and robot_state_publisher show the motion.
Succeeds the goal only after trajectory playback completes so execute() returns real SUCCESS.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

# UR5 joint names (must match move_group)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Rest position (same as joint_state_publisher zeros)
REST_POSITIONS = [0.0, -1.5707, 0.0, 0.0, 0.0, 0.0]


class FakeJointTrajectoryController(Node):
    def __init__(self):
        super().__init__("fake_joint_trajectory_controller")
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "scaled_joint_trajectory_controller/follow_joint_trajectory",
            self._execute_callback,
        )
        self._joint_state_pub = self.create_publisher(JointState, "joint_states", 10)
        # Publish rest position at startup so robot is visible
        self._publish_joint_state(REST_POSITIONS)
        self.get_logger().info(
            "Fake controller ready: follow_joint_trajectory (publishing joint_states)"
        )

    def _publish_joint_state(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(positions)
        msg.velocity = [0.0] * len(positions)
        msg.effort = []
        self._joint_state_pub.publish(msg)

    def _run_trajectory(self, trajectory):
        """Run trajectory playback in this thread (publish joint_states)."""
        if not trajectory.points:
            return
        joint_names = trajectory.joint_names
        points = trajectory.points
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        our_indices = [name_to_idx.get(name, 0) for name in JOINT_NAMES]
        rate = self.create_rate(50)
        start_time = self.get_clock().now()
        for i, point in enumerate(points):
            t = point.time_from_start.sec + 1e-9 * point.time_from_start.nanosec
            while rclpy.ok():
                elapsed = (self.get_clock().now() - start_time).nanoseconds * 1e-9
                if elapsed >= t - 0.02:
                    break
                rate.sleep()
            positions = [
                point.positions[our_indices[j]] if our_indices[j] < len(point.positions) else 0.0
                for j in range(len(JOINT_NAMES))
            ]
            self._publish_joint_state(positions)
        last = points[-1]
        positions = [
            last.positions[our_indices[j]] if our_indices[j] < len(last.positions) else 0.0
            for j in range(len(JOINT_NAMES))
        ]
        self._publish_joint_state(positions)

    def _execute_callback(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        if not trajectory.points:
            self.get_logger().warn("Empty trajectory received")
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed(result)
            return result

        self.get_logger().info(
            f"Executing trajectory: {len(trajectory.points)} points (blocking until done)"
        )
        self._run_trajectory(trajectory)
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed(result)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FakeJointTrajectoryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
