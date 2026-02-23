/**
 * UR5 KDL IK baseline node (research).
 * Loads robot model from params, uses MoveIt KDL kinematics, solves IK for a fixed pose,
 * prints joint angles. No planning, no execution.
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>

namespace {

constexpr const char* GROUP_NAME = "ur_manipulator";
constexpr const char* TIP_LINK = "tool0";
constexpr double IK_TIMEOUT_S = 5.0;
// Fixed target: position (0.4, 0.1, 0.4) m, orientation identity (tool-down)
constexpr double TARGET_X = 0.4;
constexpr double TARGET_Y = 0.1;
constexpr double TARGET_Z = 0.4;

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ur5_kdl_ik_baseline");

  // RobotModelLoader reads robot_description and robot_description_semantic from this node's params
  robot_model_loader::RobotModelLoader loader(node);
  moveit::core::RobotModelConstPtr model = loader.getModel();
  if (!model) {
    RCLCPP_ERROR(node->get_logger(), "Failed to load robot model.");
    rclcpp::shutdown();
    return 1;
  }

  const moveit::core::JointModelGroup* jmg = model->getJointModelGroup(GROUP_NAME);
  if (!jmg) {
    RCLCPP_ERROR(node->get_logger(), "Group '%s' not found.", GROUP_NAME);
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Kinematics solver: using default (KDL per kinematics.yaml).");
  RCLCPP_INFO(node->get_logger(), "Target pose: position (%.3f, %.3f, %.3f), orientation identity.", TARGET_X, TARGET_Y, TARGET_Z);

  moveit::core::RobotState state(model);
  state.setToDefaultValues();

  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = TARGET_X;
  target_pose.position.y = TARGET_Y;
  target_pose.position.z = TARGET_Z;
  target_pose.orientation.w = 1.0;
  target_pose.orientation.x = 0.0;
  target_pose.orientation.y = 0.0;
  target_pose.orientation.z = 0.0;

  bool ok = state.setFromIK(jmg, target_pose, TIP_LINK, IK_TIMEOUT_S);

  if (!ok) {
    RCLCPP_ERROR(node->get_logger(), "IK FAILED for target pose.");
    rclcpp::shutdown();
    return 1;
  }

  if (!state.satisfiesBounds(jmg)) {
    RCLCPP_ERROR(node->get_logger(), "IK returned state outside joint bounds.");
    rclcpp::shutdown();
    return 1;
  }

  std::vector<double> joint_values;
  state.copyJointGroupPositions(jmg, joint_values);
  const std::vector<std::string>& names = jmg->getActiveJointModelNames();

  RCLCPP_INFO(node->get_logger(), "IK SUCCESS.");
  RCLCPP_INFO(node->get_logger(), "Joint angles (rad):");
  for (size_t i = 0; i < names.size(); ++i) {
    RCLCPP_INFO(node->get_logger(), "  %s = %.6f", names[i].c_str(), joint_values[i]);
  }

  rclcpp::shutdown();
  return 0;
}
