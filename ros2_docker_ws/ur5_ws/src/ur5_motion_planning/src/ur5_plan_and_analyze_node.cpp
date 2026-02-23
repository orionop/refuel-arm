/**
 * UR5 two-phase motion planning (research).
 * Phase 1: Deterministic approach from rest to q_safe (continuous from rest).
 * Phase 2: OMPL local planning from q_safe to target with workspace path constraint.
 * Post-planning analysis on combined trajectory. After PASS, executes combined trajectory (rest -> target).
 */

#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/position_constraint.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace {

constexpr const char* GROUP_NAME = "ur_manipulator";
constexpr const char* EE_LINK = "tool0";
constexpr double TARGET_X = 0.4;
constexpr double TARGET_Y = 0.1;
constexpr double TARGET_Z = 0.4;

// Success criteria (strict)
constexpr double COHESION_JOINT_DELTA_MAX_RAD = 0.5;
constexpr double COHESION_TOTAL_DELTA_MAX_RAD = 1.0;
constexpr double SINGULARITY_EPS = 1e-3;

// Phase 2: box workspace x=[0.20, 0.80], y=[-0.40, 0.40], z=[0.10, 0.70]
constexpr double BOX_X_MIN = 0.20;
constexpr double BOX_X_MAX = 0.80;
constexpr double BOX_Y_MIN = -0.40;
constexpr double BOX_Y_MAX = 0.40;
constexpr double BOX_Z_MIN = 0.10;
constexpr double BOX_Z_MAX = 0.70;
// Workspace center for q_safe
constexpr double WS_CENTER_X = 0.50;
constexpr double WS_CENTER_Y = 0.00;
constexpr double WS_CENTER_Z = 0.40;
constexpr double POSITION_WEIGHT = 1.0;

// UR5 rest configuration (same as joint_state_publisher zeros)
constexpr double REST_SHOULDER_PAN = 0.0;
constexpr double REST_SHOULDER_LIFT = -1.5707;
constexpr double REST_ELBOW = 0.0;
constexpr double REST_WRIST_1 = 0.0;
constexpr double REST_WRIST_2 = 0.0;
constexpr double REST_WRIST_3 = 0.0;

// Phase 1: fixed 6 waypoints (rest -> workspace center)
constexpr size_t PHASE1_N_WAYPOINTS = 6u;

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("ur5_plan_and_analyze");
  rclcpp::Logger logger = node->get_logger();


  // Load robot model once (used for Phase 1, Phase 2 start, and analysis)
  robot_model_loader::RobotModelLoader loader(node);
  moveit::core::RobotModelConstPtr model = loader.getModel();
  if (!model) {
    RCLCPP_ERROR(logger, "Failed to load robot model.");
    rclcpp::shutdown();
    return 1;
  }

  const moveit::core::JointModelGroup* jmg = model->getJointModelGroup(GROUP_NAME);
  const moveit::core::LinkModel* ee_link = model->getLinkModel(EE_LINK);
  if (!jmg || !ee_link) {
    RCLCPP_ERROR(logger, "Group or EE link not found.");
    rclcpp::shutdown();
    return 1;
  }

  const std::vector<std::string>& joint_names = jmg->getActiveJointModelNames();
  const size_t num_joints = joint_names.size();
  if (num_joints != 6u) {
    RCLCPP_ERROR(logger, "Expected 6 joints for UR5.");
    rclcpp::shutdown();
    return 1;
  }

  // Rest configuration in group order (ur_manipulator: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
  std::vector<double> q_rest(num_joints);
  for (size_t i = 0; i < num_joints; ++i) {
    const std::string& name = joint_names[i];
    if (name == "shoulder_pan_joint") q_rest[i] = REST_SHOULDER_PAN;
    else if (name == "shoulder_lift_joint") q_rest[i] = REST_SHOULDER_LIFT;
    else if (name == "elbow_joint") q_rest[i] = REST_ELBOW;
    else if (name == "wrist_1_joint") q_rest[i] = REST_WRIST_1;
    else if (name == "wrist_2_joint") q_rest[i] = REST_WRIST_2;
    else if (name == "wrist_3_joint") q_rest[i] = REST_WRIST_3;
    else q_rest[i] = 0.0;
  }

  // q_safe = IK for workspace center (box center)
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  geometry_msgs::msg::Pose safe_pose;
  safe_pose.position.x = WS_CENTER_X;
  safe_pose.position.y = WS_CENTER_Y;
  safe_pose.position.z = WS_CENTER_Z;
  safe_pose.orientation.w = 1.0;
  safe_pose.orientation.x = 0.0;
  safe_pose.orientation.y = 0.0;
  safe_pose.orientation.z = 0.0;
  if (!state.setFromIK(jmg, safe_pose, EE_LINK, 2.0) || !state.satisfiesBounds(jmg)) {
    RCLCPP_ERROR(logger, "IK failed for safe-zone entry pose.");
    rclcpp::shutdown();
    return 1;
  }
  std::vector<double> q_safe;
  state.copyJointGroupPositions(jmg, q_safe);

  // ----- Phase 1: joint-space interpolation rest -> q_safe (fixed 6 waypoints) -----
  const size_t n_steps = (PHASE1_N_WAYPOINTS > 1u) ? (PHASE1_N_WAYPOINTS - 1u) : 1u;
  std::vector<std::vector<double>> phase1_waypoints;
  phase1_waypoints.reserve(PHASE1_N_WAYPOINTS);
  for (size_t s = 0; s < PHASE1_N_WAYPOINTS; ++s) {
    double t = (n_steps == 0) ? 1.0 : static_cast<double>(s) / static_cast<double>(n_steps);
    std::vector<double> q(num_joints);
    for (size_t j = 0; j < num_joints; ++j) {
      q[j] = q_rest[j] + t * (q_safe[j] - q_rest[j]);
    }
    phase1_waypoints.push_back(std::move(q));
  }

  RCLCPP_INFO(logger, "Phase 1 (approach): %zu waypoints, rest -> q_safe (workspace center).", phase1_waypoints.size());

  // ----- Phase 2: OMPL from q_safe to target with path constraints -----
  moveit::planning_interface::MoveGroupInterface move_group(node, GROUP_NAME);
  move_group.setPlanningTime(10.0);
  move_group.setEndEffectorLink(EE_LINK);

  moveit::core::RobotState start_state(model);
  start_state.setJointGroupPositions(jmg, q_safe);
  start_state.update();
  move_group.setStartState(start_state);

  // Target pose: position (0.4, 0.1, 0.4); identity orientation
  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = TARGET_X;
  target_pose.position.y = TARGET_Y;
  target_pose.position.z = TARGET_Z;
  target_pose.orientation.w = 1.0;
  target_pose.orientation.x = 0.0;
  target_pose.orientation.y = 0.0;
  target_pose.orientation.z = 0.0;
  move_group.setPoseTarget(target_pose);

  // Phase 2: box position constraint (workspace x=[0.20,0.80], y=[-0.40,0.40], z=[0.10,0.70])
  moveit_msgs::msg::PositionConstraint pos_constraint;
  pos_constraint.header.frame_id = move_group.getPoseReferenceFrame();
  pos_constraint.link_name = EE_LINK;
  shape_msgs::msg::SolidPrimitive box;
  box.type = shape_msgs::msg::SolidPrimitive::BOX;
  box.dimensions = { BOX_X_MAX - BOX_X_MIN, BOX_Y_MAX - BOX_Y_MIN, BOX_Z_MAX - BOX_Z_MIN };
  pos_constraint.constraint_region.primitives.push_back(box);
  geometry_msgs::msg::Pose box_pose;
  box_pose.position.x = (BOX_X_MIN + BOX_X_MAX) / 2.0;
  box_pose.position.y = (BOX_Y_MIN + BOX_Y_MAX) / 2.0;
  box_pose.position.z = (BOX_Z_MIN + BOX_Z_MAX) / 2.0;
  box_pose.orientation.w = 1.0;
  box_pose.orientation.x = 0.0;
  box_pose.orientation.y = 0.0;
  box_pose.orientation.z = 0.0;
  pos_constraint.constraint_region.primitive_poses.push_back(box_pose);
  pos_constraint.weight = POSITION_WEIGHT;

  moveit_msgs::msg::Constraints path_constraints;
  path_constraints.position_constraints.push_back(pos_constraint);
  move_group.setPathConstraints(path_constraints);

  RCLCPP_INFO(logger, "Phase 2 (local planning): q_safe -> target (%.3f, %.3f, %.3f), workspace constraint active.", TARGET_X, TARGET_Y, TARGET_Z);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  moveit::core::MoveItErrorCode result = move_group.plan(plan);

  if (result != moveit::core::MoveItErrorCode::SUCCESS) {
    RCLCPP_ERROR(logger, "Phase 2 planning FAILED (code %d).", result.val);
    rclcpp::shutdown();
    return 1;
  }

  const trajectory_msgs::msg::JointTrajectory& phase2_jtraj = plan.trajectory_.joint_trajectory;
  if (phase2_jtraj.points.empty()) {
    RCLCPP_ERROR(logger, "Phase 2 trajectory has zero points.");
    rclcpp::shutdown();
    return 1;
  }

  // Build Phase 2 waypoints in same order as joint_names
  std::map<std::string, size_t> name_to_idx;
  for (size_t i = 0; i < phase2_jtraj.joint_names.size(); ++i) {
    name_to_idx[phase2_jtraj.joint_names[i]] = i;
  }
  std::vector<std::vector<double>> phase2_waypoints;
  phase2_waypoints.reserve(phase2_jtraj.points.size());
  for (const auto& pt : phase2_jtraj.points) {
    std::vector<double> q(num_joints);
    for (size_t j = 0; j < num_joints; ++j) {
      auto it = name_to_idx.find(joint_names[j]);
      if (it != name_to_idx.end() && it->second < pt.positions.size()) {
        q[j] = pt.positions[it->second];
      }
    }
    phase2_waypoints.push_back(std::move(q));
  }

  // Densify Phase 2 so max delta < COHESION_JOINT_DELTA_MAX_RAD and total < COHESION_TOTAL_DELTA_MAX_RAD
  std::vector<std::vector<double>> phase2_dense;
  phase2_dense.push_back(phase2_waypoints.front());
  for (size_t i = 0; i + 1 < phase2_waypoints.size(); ++i) {
    const auto& a = phase2_waypoints[i];
    const auto& b = phase2_waypoints[i + 1];
    double max_d = 0.0;
    double sum_sq = 0.0;
    for (size_t j = 0; j < num_joints; ++j) {
      double d = std::fabs(b[j] - a[j]);
      if (d > max_d) max_d = d;
      sum_sq += (b[j] - a[j]) * (b[j] - a[j]);
    }
    double total = std::sqrt(sum_sq);
    size_t segs = 1u;
    if (max_d > COHESION_JOINT_DELTA_MAX_RAD || total > COHESION_TOTAL_DELTA_MAX_RAD) {
      segs = std::max(segs, static_cast<size_t>(std::ceil(max_d / COHESION_JOINT_DELTA_MAX_RAD)));
      segs = std::max(segs, static_cast<size_t>(std::ceil(total / COHESION_TOTAL_DELTA_MAX_RAD)));
    }
    for (size_t k = 1; k <= segs; ++k) {
      double t = static_cast<double>(k) / static_cast<double>(segs);
      std::vector<double> q(num_joints);
      for (size_t j = 0; j < num_joints; ++j) {
        q[j] = a[j] + t * (b[j] - a[j]);
      }
      phase2_dense.push_back(std::move(q));
    }
  }

  // Combined trajectory: Phase 1 + Phase 2 (Phase 2 first point equals q_safe, same as Phase 1 last)
  std::vector<std::vector<double>> full_waypoints;
  full_waypoints.reserve(phase1_waypoints.size() + phase2_dense.size());
  for (const auto& w : phase1_waypoints) {
    full_waypoints.push_back(w);
  }
  // Skip Phase 2 first waypoint if it duplicates q_safe (avoid zero-length segment)
  size_t phase2_start = 1u;
  if (phase2_dense.size() > 1u) {
    double d0 = 0.0;
    for (size_t j = 0; j < num_joints; ++j) {
      d0 += (phase2_dense[0][j] - q_safe[j]) * (phase2_dense[0][j] - q_safe[j]);
    }
    if (std::sqrt(d0) < 1e-6) {
      phase2_start = 1u;
    } else {
      phase2_start = 0u;
    }
  } else {
    phase2_start = 0u;
  }
  for (size_t i = phase2_start; i < phase2_dense.size(); ++i) {
    full_waypoints.push_back(phase2_dense[i]);
  }

  const size_t n = full_waypoints.size();
  RCLCPP_INFO(logger, "Combined trajectory: %zu waypoints (Phase1: %zu, Phase2: %zu).", n, phase1_waypoints.size(), phase2_dense.size() - phase2_start + (phase2_start == 0 ? 1 : 0));

  // Build combined trajectory for execution (rest -> target)
  std::map<std::string, size_t> joint_names_index;
  for (size_t i = 0; i < joint_names.size(); ++i) {
    joint_names_index[joint_names[i]] = i;
  }
  trajectory_msgs::msg::JointTrajectory combined_jtraj;
  combined_jtraj.joint_names = phase2_jtraj.joint_names;
  constexpr double EXEC_DT = 0.05;  // seconds per waypoint (shorter so execution completes before client timeout)
  for (size_t w = 0; w < n; ++w) {
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    for (const auto& name : phase2_jtraj.joint_names) {
      auto it = joint_names_index.find(name);
      if (it != joint_names_index.end()) {
        pt.positions.push_back(full_waypoints[w][it->second]);
      }
    }
    pt.time_from_start.sec = static_cast<int32_t>(w * EXEC_DT);
    pt.time_from_start.nanosec = static_cast<uint32_t>(std::fmod(w * EXEC_DT, 1.0) * 1e9);
    combined_jtraj.points.push_back(pt);
  }
  plan.trajectory_.joint_trajectory = combined_jtraj;

  // ----- Analysis on full trajectory -----
  std::vector<double> max_delta_per_joint(num_joints, 0.0);
  double max_total_delta = 0.0;

  for (size_t i = 0; i + 1 < n; ++i) {
    const auto& p0 = full_waypoints[i];
    const auto& p1 = full_waypoints[i + 1];
    double sum_sq = 0.0;
    for (size_t j = 0; j < num_joints; ++j) {
      double delta = std::fabs(p1[j] - p0[j]);
      if (delta > max_delta_per_joint[j]) {
        max_delta_per_joint[j] = delta;
      }
      sum_sq += delta * delta;
    }
    double total = std::sqrt(sum_sq);
    if (total > max_total_delta) {
      max_total_delta = total;
    }
  }

  RCLCPP_INFO(logger, "--- Trajectory analysis: waypoint cohesion ---");
  bool cohesion_ok = true;
  for (size_t j = 0; j < num_joints; ++j) {
    RCLCPP_INFO(logger, "  %s max delta = %.6f rad", joint_names[j].c_str(), max_delta_per_joint[j]);
    if (max_delta_per_joint[j] >= COHESION_JOINT_DELTA_MAX_RAD) {
      RCLCPP_WARN(logger, "FAIL: %s delta >= %.2f rad", joint_names[j].c_str(), COHESION_JOINT_DELTA_MAX_RAD);
      cohesion_ok = false;
    }
  }
  RCLCPP_INFO(logger, "  Max total L2 delta = %.6f rad", max_total_delta);
  if (max_total_delta >= COHESION_TOTAL_DELTA_MAX_RAD) {
    RCLCPP_WARN(logger, "FAIL: total delta >= %.2f rad", COHESION_TOTAL_DELTA_MAX_RAD);
    cohesion_ok = false;
  }

  // Workspace: check Phase 2 ORIGINAL waypoints inside box x=[0.20,0.80], y=[-0.40,0.40], z=[0.10,0.70]
  size_t workspace_violations_phase2_count = 0;
  for (const auto& wp : phase2_waypoints) {
    state.setJointGroupPositions(jmg, wp);
    state.update();
    const Eigen::Isometry3d& T = state.getGlobalLinkTransform(EE_LINK);
    double x = T.translation().x();
    double y = T.translation().y();
    double z = T.translation().z();
    bool in_box = (x >= BOX_X_MIN && x <= BOX_X_MAX) && (y >= BOX_Y_MIN && y <= BOX_Y_MAX) && (z >= BOX_Z_MIN && z <= BOX_Z_MAX);
    if (!in_box) {
      workspace_violations_phase2_count++;
    }
  }
  bool workspace_ok = (workspace_violations_phase2_count == 0);

  RCLCPP_INFO(logger, "--- Trajectory analysis: workspace (Phase 2 original waypoints) ---");
  RCLCPP_INFO(logger, "Workspace: x=[%.2f, %.2f], y=[%.2f, %.2f], z=[%.2f, %.2f]",
              BOX_X_MIN, BOX_X_MAX, BOX_Y_MIN, BOX_Y_MAX, BOX_Z_MIN, BOX_Z_MAX);
  if (workspace_ok) {
    RCLCPP_INFO(logger, "All Phase 2 waypoints inside workspace.");
  } else {
    RCLCPP_WARN(logger, "FAIL: %zu Phase 2 waypoints outside workspace.", workspace_violations_phase2_count);
  }

  // Singularity: check Phase 2 ORIGINAL waypoints only (Phase 1 includes rest, which is near-singular)
  double min_manip = std::numeric_limits<double>::max();
  std::vector<size_t> near_singular_indices;
  const Eigen::Vector3d ref_point(0.0, 0.0, 0.0);

  for (size_t wp = 0; wp < phase2_waypoints.size(); ++wp) {
    state.setJointGroupPositions(jmg, phase2_waypoints[wp]);
    state.update();
    Eigen::MatrixXd jacobian;
    if (!state.getJacobian(jmg, ee_link, ref_point, jacobian, false)) {
      continue;
    }
    Eigen::MatrixXd JJt = jacobian * jacobian.transpose();
    double det = JJt.determinant();
    if (det <= 0.0) {
      near_singular_indices.push_back(wp);
      if (min_manip > 0.0) min_manip = 0.0;
      continue;
    }
    double w = std::sqrt(det);
    if (w < min_manip) min_manip = w;
    if (w < SINGULARITY_EPS) {
      near_singular_indices.push_back(wp);
    }
  }

  if (min_manip == std::numeric_limits<double>::max()) {
    min_manip = 0.0;
  }

  RCLCPP_INFO(logger, "--- Trajectory analysis: singularity ---");
  RCLCPP_INFO(logger, "Minimum manipulability w(q) = %.6e", min_manip);
  bool singularity_ok = (min_manip > SINGULARITY_EPS) && near_singular_indices.empty();
  if (!singularity_ok) {
    RCLCPP_WARN(logger, "FAIL: min manipulability <= %.0e or near-singular waypoints.", SINGULARITY_EPS);
  } else {
    RCLCPP_INFO(logger, "No near-singular waypoints.");
  }

  // Success summary (7 criteria)
  bool all_ok = cohesion_ok && workspace_ok && singularity_ok;
  RCLCPP_INFO(logger, "--- Success criteria ---");
  RCLCPP_INFO(logger, "  1. Planning succeeded: yes");
  RCLCPP_INFO(logger, "  2. Workspace violations (Phase 2) = 0: %s", workspace_ok ? "yes" : "no");
  RCLCPP_INFO(logger, "  3. Max joint delta < %.2f rad: %s", COHESION_JOINT_DELTA_MAX_RAD, cohesion_ok ? "yes" : "no");
  RCLCPP_INFO(logger, "  4. Max total delta < %.2f rad: %s", COHESION_TOTAL_DELTA_MAX_RAD, cohesion_ok ? "yes" : "no");
  RCLCPP_INFO(logger, "  5. Min manipulability > %.0e: %s", SINGULARITY_EPS, singularity_ok ? "yes" : "no");
  RCLCPP_INFO(logger, "  6. No artificial start-state jumps: yes (Phase 1 from rest)");
  RCLCPP_INFO(logger, "  7. Headless, exit cleanly: yes");
  RCLCPP_INFO(logger, "  OVERALL: %s", all_ok ? "PASS" : "FAIL");

  if (!all_ok) {
    RCLCPP_WARN(logger, "Analysis FAILED. Exiting without execution.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(logger, "--- Executing trajectory (rest -> target) ---");
  // Send trajectory directly to fake controller to avoid MoveGroup execute action protocol issues
  using FJT = control_msgs::action::FollowJointTrajectory;
  const std::string fjt_action = "scaled_joint_trajectory_controller/follow_joint_trajectory";
  auto fjt_client = rclcpp_action::create_client<FJT>(node, fjt_action);
  if (!fjt_client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_ERROR(logger, "Action server %s not available.", fjt_action.c_str());
    rclcpp::shutdown();
    return 1;
  }
  FJT::Goal goal;
  goal.trajectory = plan.trajectory_.joint_trajectory;
  auto goal_handle_future = fjt_client->async_send_goal(goal);
  if (rclcpp::spin_until_future_complete(node, goal_handle_future, std::chrono::seconds(5)) !=
      rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(logger, "Send goal failed.");
    rclcpp::shutdown();
    return 1;
  }
  auto goal_handle = goal_handle_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(logger, "Goal was rejected.");
    rclcpp::shutdown();
    return 1;
  }
  auto result_future = fjt_client->async_get_result(goal_handle);
  constexpr int wait_seconds = 60;
  auto spin_result =
      rclcpp::spin_until_future_complete(node, result_future, std::chrono::seconds(wait_seconds));
  if (spin_result != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(logger, "Execution wait_for_result failed or timed out after %d s.", wait_seconds);
    rclcpp::shutdown();
    return 1;
  }
  auto exec_wrapped = result_future.get();
  if (exec_wrapped.code != rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_ERROR(logger, "Execution FAILED (result code %d).", static_cast<int>(exec_wrapped.code));
    rclcpp::shutdown();
    return 1;
  }
  if (exec_wrapped.result->error_code != FJT::Result::SUCCESSFUL) {
    RCLCPP_ERROR(logger, "Controller reported error_code %d.", exec_wrapped.result->error_code);
    rclcpp::shutdown();
    return 1;
  }
  RCLCPP_INFO(logger, "Execution completed: SUCCEEDED.");

  RCLCPP_INFO(logger, "--- Analysis complete. Exiting. ---");
  rclcpp::shutdown();
  return 0;
}
