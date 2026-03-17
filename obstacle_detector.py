#!/usr/bin/env python3
"""
Obstacle Detector: Simulated sensor via Gazebo /model_states
=============================================================

Subscribes to /gazebo/model_states and treats any model NOT in the
whitelist as a potential obstacle.  Detection is range-gated: obstacles
only become "known" when any FK checkpoint of the arm is within
SENSOR_RANGE meters.

Output format: list of (center_xyz, radius) tuples — directly compatible
with stomp_collision.stomp_optimize() and elastic_strips.elastic_strip_deform().
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))

from elastic_strips import fk_checkpoints

# ── Constants ─────────────────────────────────────────────────────

KNOWN_MODELS = {
    'ground_plane', 'sun',
    'kr6_r700',             # the robot itself
    'refuel_car',           # car model
    'refuel_platform',      # elevated stand
    'fuel_inlet_marker',    # green inlet overlay
}

SENSOR_RANGE     = 0.6     # metres — detection radius
DEFAULT_OBS_RADIUS = 0.08  # conservative enclosing sphere for unknown objects

# Default demo obstacle (spawned at startup for reliable demos)
DEFAULT_OBSTACLE_CENTER = np.array([0.58, 0.23, 0.73])
DEFAULT_OBSTACLE_RADIUS = 0.05

DEFAULT_OBSTACLE_SDF = f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="obstacle_default">
    <static>true</static>
    <link name="link">
      <visual name="vis">
        <geometry><sphere><radius>{DEFAULT_OBSTACLE_RADIUS}</radius></sphere></geometry>
        <material><ambient>0 0 1 1</ambient><diffuse>0 0.1 1 1</diffuse></material>
      </visual>
      <collision name="col">
        <geometry><sphere><radius>{DEFAULT_OBSTACLE_RADIUS}</radius></sphere></geometry>
      </collision>
    </link>
  </model>
</sdf>"""


# ── Gazebo-backed Detector ────────────────────────────────────────

class ObstacleDetector:
    """Real-time obstacle detection via /gazebo/model_states."""

    def __init__(self, sensor_range=SENSOR_RANGE):
        import rospy
        from gazebo_msgs.msg import ModelStates

        self.sensor_range = sensor_range
        self._latest = None
        self._detected = {}          # model_name → (center, radius)
        self._prev_count = 0

        self._sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates,
            self._cb, queue_size=1)

    def _cb(self, msg):
        self._latest = msg

    def update(self, q_current):
        """Check for new obstacles within sensor range of arm checkpoints."""
        if self._latest is None:
            return

        arm_pts = [p for _, p in fk_checkpoints(q_current)]
        msg = self._latest

        for name, pose in zip(msg.name, msg.pose):
            if name in KNOWN_MODELS or name in self._detected:
                continue

            obs_pos = np.array([pose.position.x,
                                pose.position.y,
                                pose.position.z])

            for arm_p in arm_pts:
                if np.linalg.norm(arm_p - obs_pos) < self.sensor_range:
                    self._detected[name] = (obs_pos, DEFAULT_OBS_RADIUS)
                    print(f"     Obstacle DETECTED: '{name}' at "
                          f"[{obs_pos[0]:.2f}, {obs_pos[1]:.2f}, {obs_pos[2]:.2f}]")
                    break

    def get_obstacles(self):
        """Return detected obstacles as [(center, radius), ...]."""
        return list(self._detected.values())

    def has_new(self):
        """True if new obstacles detected since last call."""
        changed = len(self._detected) > self._prev_count
        self._prev_count = len(self._detected)
        return changed


def spawn_default_obstacle():
    """Spawn the default demo obstacle (blue sphere) in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    p = Pose()
    p.position.x = DEFAULT_OBSTACLE_CENTER[0]
    p.position.y = DEFAULT_OBSTACLE_CENTER[1]
    p.position.z = DEFAULT_OBSTACLE_CENTER[2]
    p.orientation.w = 1.0
    try:
        spawn("obstacle_default", DEFAULT_OBSTACLE_SDF, "/", p, "world")
        print(f"  Spawned default obstacle at "
              f"[{DEFAULT_OBSTACLE_CENTER[0]:.2f}, "
              f"{DEFAULT_OBSTACLE_CENTER[1]:.2f}, "
              f"{DEFAULT_OBSTACLE_CENTER[2]:.2f}]")
    except Exception as e:
        print(f"  Default obstacle spawn note: {e}")


# ── Non-ROS Fallback ──────────────────────────────────────────────

class DummyDetector:
    """Offline obstacle detector for --rviz / dry-run modes.

    Simulates range-gated discovery using hardcoded obstacles.
    """
    def __init__(self, obstacles=None, sensor_range=SENSOR_RANGE):
        self.sensor_range = sensor_range
        self._pending = list(obstacles or [
            (DEFAULT_OBSTACLE_CENTER.copy(), DEFAULT_OBSTACLE_RADIUS)
        ])
        self._detected = []
        self._prev_count = 0

    def update(self, q_current):
        arm_pts = [p for _, p in fk_checkpoints(q_current)]
        still_pending = []
        for center, radius in self._pending:
            found = False
            for arm_p in arm_pts:
                if np.linalg.norm(arm_p - center) < self.sensor_range:
                    self._detected.append((center, radius))
                    print(f"     Obstacle DETECTED (sim): "
                          f"[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
                    found = True
                    break
            if not found:
                still_pending.append((center, radius))
        self._pending = still_pending

    def get_obstacles(self):
        return list(self._detected)

    def has_new(self):
        changed = len(self._detected) > self._prev_count
        self._prev_count = len(self._detected)
        return changed
