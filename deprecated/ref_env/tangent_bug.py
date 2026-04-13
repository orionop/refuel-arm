#!/usr/bin/env python3
"""
Tangent Bug Algorithm (Kamon, Rimon, Rivlin 1998)
================================================
A sensor-based motion planner that uses a local tangent graph (LTG)
to navigate towards a goal while avoiding unknown obstacles.

Modes:
  1. Motion-to-Goal (MtG): Move toward the point on the LTG that 
     minimizes the distance to the goal.
  2. Boundary-Following (BF): Follow the boundary of an obstacle
     until a 'leave' condition is met.

Adapted for a 6-DOF arm end-effector in Cartesian workspace.
"""
import numpy as np
from ik_geometric import fwd_kinematics, KIN_KR6_R700, KIN_UR5 # type: ignore


def _is_kuka(kin):
    """Determine robot type from kinematic parameters (avoids fragile dict ==)."""
    if kin is None:
        return True
    h1 = np.asarray(kin.get('H', KIN_KR6_R700['H']))
    return h1[2, 0] < 0  # KUKA H1 z-component is -1; UR5 is +1


class TangentBugPlanner:
    def __init__(self, q_start, q_goal, obstacles, sensor_range=0.5, kin=None):
        self.q_start = q_start
        self.q_goal = q_goal
        self.obstacles = obstacles
        self.Rs = sensor_range
        self.kin = kin if kin is not None else KIN_KR6_R700
        
        _, self.p_start = fwd_kinematics(q_start, kin=self.kin)
        _, self.p_goal = fwd_kinematics(q_goal, kin=self.kin)
        
        self.state = "MtG"
        self.d_min = float('inf') # Minimum distance to goal seen so far on boundary
        self.boundary_start_p = None
        self.boundary_hit_p = None
        
    def _get_distance_to_obs(self, p, obs):
        """Analytical distance to obstacle surface."""
        obs_type, center, dims, yaw = obs
        if obs_type == 'sphere':
            return np.linalg.norm(p - center) - dims
        elif obs_type == 'box':
            # Simplified box distance (AABB approach for sensor)
            R_rot = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0,0,1]])
            pl = R_rot @ (p - center)
            dx = max(0, abs(pl[0]) - dims[0]/2)
            dy = max(0, abs(pl[1]) - dims[1]/2)
            dz = max(0, abs(pl[2]) - dims[2]/2)
            return np.sqrt(dx**2 + dy**2 + dz**2)
        elif obs_type == 'cylinder':
            d_xy = max(0, np.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2) - dims[0])
            d_z = max(0, abs(p[2]-center[2]) - dims[1]/2)
            return np.sqrt(d_xy**2 + d_z**2)
        return 1.0

    def sense(self, p):
        """Simulate a 360-degree range sensor. 
        Returns a set of 'detected' boundary points within Rs.
        In a real robot, this would be a LIDAR/depth scan.
        """
        detected_points = []
        # Sample points around the obstacle boundaries
        for obs in self.obstacles:
            # Check if any part of the obstacle is within Rs
            d_center = np.linalg.norm(p - obs[1])
            if d_center < self.Rs + 1.0: # Broad phase
                # Find the closest point on this obstacle
                # (Simple approximation: vector from p to center intersected with surface)
                vec = obs[1] - p
                dist = self._get_distance_to_obs(p, obs)
                if dist < self.Rs:
                    closest_pt = p + (vec / (np.linalg.norm(vec)+1e-9)) * dist
                    detected_points.append(closest_pt)
        return detected_points

    def plan_step(self, p_curr):
        """Compute the next workspace velocity vector."""
        dist_to_goal = np.linalg.norm(self.p_goal - p_curr)
        if dist_to_goal < 0.02:
            return np.zeros(3) # Reached

        detected = self.sense(p_curr)
        
        # 1. Check for blocking obstacles in the direct goal direction
        goal_vec = (self.p_goal - p_curr) / (dist_to_goal + 1e-9)
        blocked = False
        for pt in detected:
            vec_to_pt = pt - p_curr
            dist_to_pt = np.linalg.norm(vec_to_pt)
            # If a detected point is very close and in front
            if dist_to_pt < 0.1 and np.dot(goal_vec, vec_to_pt/dist_to_pt) > 0.8:
                blocked = True
                break

        if self.state == "MtG":
            if not blocked:
                # Move directly to goal
                return goal_vec
            else:
                self.state = "BF"
                self.boundary_hit_p = p_curr
                self.d_min = dist_to_goal
                return self._follow_boundary(p_curr, detected)
        
        elif self.state == "BF":
            # Leave BF if we find a point with distance to goal < d_min
            if dist_to_goal < self.d_min - 0.05:
                self.state = "MtG"
                return goal_vec
            return self._follow_boundary(p_curr, detected)

    def _follow_boundary(self, p_curr, detected):
        """3D wall-following: tangent is the goal direction projected onto the obstacle surface."""
        if not detected: return (self.p_goal - p_curr) / (np.linalg.norm(self.p_goal - p_curr)+1e-9)

        # Closest detected point defines the outward normal
        closest = detected[0]
        normal = (p_curr - closest) / (np.linalg.norm(p_curr - closest) + 1e-9)

        # Project goal direction onto the plane perpendicular to the normal
        goal_dir = (self.p_goal - p_curr)
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-9)
        tangent = goal_dir - np.dot(goal_dir, normal) * normal
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-6:
            # Goal is directly behind the obstacle — pick an arbitrary perpendicular
            # Use cross product with the least-aligned basis vector
            abs_n = np.abs(normal)
            axis = np.array([1., 0., 0.]) if abs_n[0] < abs_n[2] else np.array([0., 0., 1.])
            tangent = np.cross(normal, axis)
            t_norm = np.linalg.norm(tangent)
        tangent = tangent / (t_norm + 1e-9)

        # Stay at a fixed 'safe' distance from boundary
        SAFE_DIST = 0.15
        d = np.linalg.norm(p_curr - closest)
        correction = normal * (SAFE_DIST - d)

        result = tangent + 0.5 * correction
        return result / (np.linalg.norm(result) + 1e-9)

def tangent_bug_optimize(q_start, q_goal, obstacles, max_steps=100, step_size=0.05, sensor_range=0.5, kin=None):
    """Generates a full trajectory using Tangent Bug logic."""
    planner = TangentBugPlanner(q_start, q_goal, obstacles, sensor_range, kin)
    trajectory = [q_start.copy()]
    curr_q = q_start.copy()
    
    from ik_geometric import IK_solve # type: ignore
    
    for _ in range(max_steps):
        _, p_curr = fwd_kinematics(curr_q, kin=kin)
        if np.linalg.norm(planner.p_goal - p_curr) < 0.03:
            break
            
        v = planner.plan_step(p_curr)
        if np.linalg.norm(v) < 1e-3: break
        
        p_next = p_curr + v * step_size
        # Use simple IK to find next q (incremental)
        # Note: Tangent Bug usually works for mobile robots, here we map to arm EE
        # We assume constant orientation (pointing at goal?)
        # For simplicity, we use the goal orientation
        R_goal, _ = fwd_kinematics(q_goal, kin=kin)
        Q_sol = IK_solve(R_goal, p_next, robot="kuka" if _is_kuka(kin) else "ur5")
        
        if Q_sol.size > 0:
            # Pick solution closest to current q
            diffs = Q_sol - curr_q[:, np.newaxis]
            idx = np.argmin(np.linalg.norm(diffs, axis=0))
            curr_q = Q_sol[:, idx]
            trajectory.append(curr_q.copy())
        else:
            print("TangentBug: IK failing during boundary follow!")
            break
            
    return np.array(trajectory)
