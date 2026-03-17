#!/usr/bin/env python3
"""
Car Model: SDF geometry, Gazebo spawning, and fuel inlet pose computation
=========================================================================

Spawns a simple box-car on an elevated platform in Gazebo and computes
the fuel inlet target pose for the KUKA KR6 R700 end-effector.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'kuka_refuel_ws', 'src',
    'kuka_kr6_gazebo', 'scripts')))
from ik_geometric import rot

# ── Geometry Constants ────────────────────────────────────────────
PLATFORM_HEIGHT = 0.35                          # Elevated stand (m)
PLATFORM_SIZE   = (0.60, 0.40, PLATFORM_HEIGHT) # L x W x H

CAR_BODY_SIZE   = (0.50, 0.25, 0.12)           # L x W x H  (simplified sedan)
WHEEL_RADIUS    = 0.035
WHEEL_LENGTH    = 0.02

INLET_SIZE      = (0.06, 0.005, 0.06)          # visible green square on car side

# Car center sits on top of the platform
CAR_POSITION_DEFAULT = np.array([0.55, 0.35, PLATFORM_HEIGHT + CAR_BODY_SIZE[2] / 2])
CAR_YAW_DEFAULT      = 0.0

# Inlet offset in the car's local frame  (-Y side = toward robot)
INLET_OFFSET_LOCAL = np.array([0.0, -(CAR_BODY_SIZE[1] / 2 + 0.002), 0.02])

# EE orientation: tool axis pointing +Y (into the car's -Y face)
# Col 0 = [0,1,0] (tool→+Y), Col 1 = [0,0,1] (up), Col 2 = [1,0,0] (right)
R_TOOL_INTO_CAR = np.array([
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.]])


# ── SDF Templates ─────────────────────────────────────────────────

PLATFORM_SDF = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="refuel_platform">
    <static>true</static>
    <link name="link">
      <visual name="vis">
        <geometry><box><size>{pl} {pw} {ph}</size></box></geometry>
        <material>
          <ambient>0.45 0.45 0.45 1</ambient>
          <diffuse>0.50 0.50 0.50 1</diffuse>
        </material>
      </visual>
      <collision name="col">
        <geometry><box><size>{pl} {pw} {ph}</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>""".format(pl=PLATFORM_SIZE[0], pw=PLATFORM_SIZE[1], ph=PLATFORM_SIZE[2])


def _car_sdf():
    """Build the inline SDF for the car body + wheels + inlet panel."""
    bx, by, bz = CAR_BODY_SIZE
    wr, wl = WHEEL_RADIUS, WHEEL_LENGTH
    ix, iy, iz = INLET_SIZE

    # Wheel positions relative to body center
    wx = bx / 2 - 0.06
    wy = by / 2 + wl / 2 + 0.001
    wz = -(bz / 2)

    wheel_block = ""
    for idx, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        wheel_block += f"""
      <link name="wheel_{idx}">
        <pose>{sx*wx} {sy*wy} {wz} 1.5708 0 0</pose>
        <visual name="vis"><geometry><cylinder><radius>{wr}</radius><length>{wl}</length></cylinder></geometry>
          <material><ambient>0.15 0.15 0.15 1</ambient><diffuse>0.20 0.20 0.20 1</diffuse></material>
        </visual>
      </link>"""

    # Inlet on -Y face
    iy_pos = -(by / 2)
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="refuel_car">
    <static>true</static>
    <link name="body">
      <visual name="vis">
        <geometry><box><size>{bx} {by} {bz}</size></box></geometry>
        <material><ambient>0.55 0.55 0.60 1</ambient><diffuse>0.60 0.60 0.65 1</diffuse></material>
      </visual>
      <collision name="col">
        <geometry><box><size>{bx} {by} {bz}</size></box></geometry>
      </collision>
    </link>
    <link name="inlet">
      <pose>0 {iy_pos} 0.02 0 0 0</pose>
      <visual name="vis">
        <geometry><box><size>{ix} {iy} {iz}</size></box></geometry>
        <material><ambient>0.0 0.85 0.0 1</ambient><diffuse>0.0 0.90 0.0 1</diffuse></material>
      </visual>
    </link>{wheel_block}
  </model>
</sdf>"""
    return sdf

CAR_SDF = _car_sdf()


# ── Pose Computation ──────────────────────────────────────────────

def get_inlet_pose(car_pos, car_yaw=0.0):
    """Compute world-frame inlet position and EE orientation.

    Returns (inlet_xyz, inlet_R) where inlet_R orients the EE tool axis
    perpendicular to the inlet face (pointing into the car).
    """
    R_yaw = rot(np.array([0., 0., 1.]), car_yaw)
    inlet_xyz = car_pos + R_yaw @ INLET_OFFSET_LOCAL

    # Rotate the base EE orientation by the car's yaw
    inlet_R = R_yaw @ R_TOOL_INTO_CAR

    return inlet_xyz, inlet_R


def get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.08):
    """Pull the target back from the inlet along the approach normal.

    The approach direction is the first column of inlet_R (tool axis).
    """
    approach_dir = inlet_R[:, 0]  # tool axis direction
    preapproach_xyz = inlet_xyz - standoff * approach_dir
    return preapproach_xyz, inlet_R


# ── Gazebo Spawning ───────────────────────────────────────────────

def spawn_car_gazebo(car_pos=None, car_yaw=None):
    """Spawn the elevated platform and car model in Gazebo."""
    import rospy
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose
    import tf.transformations as tft

    if car_pos is None:
        car_pos = CAR_POSITION_DEFAULT
    if car_yaw is None:
        car_yaw = CAR_YAW_DEFAULT

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

    # 1. Platform
    p_plat = Pose()
    p_plat.position.x = car_pos[0]
    p_plat.position.y = car_pos[1]
    p_plat.position.z = PLATFORM_HEIGHT / 2
    p_plat.orientation.w = 1.0
    try:
        spawn("refuel_platform", PLATFORM_SDF, "/", p_plat, "world")
    except Exception as e:
        print(f"  Platform spawn note: {e}")

    # 2. Car on top of platform
    p_car = Pose()
    p_car.position.x = car_pos[0]
    p_car.position.y = car_pos[1]
    p_car.position.z = car_pos[2]
    quat = tft.quaternion_from_euler(0, 0, car_yaw)
    p_car.orientation.x = quat[0]
    p_car.orientation.y = quat[1]
    p_car.orientation.z = quat[2]
    p_car.orientation.w = quat[3]
    try:
        spawn("refuel_car", CAR_SDF, "/", p_car, "world")
    except Exception as e:
        print(f"  Car spawn note: {e}")

    print(f"  Spawned platform + car at [{car_pos[0]:.2f}, {car_pos[1]:.2f}, {car_pos[2]:.2f}]")


# ── RViz Markers ──────────────────────────────────────────────────

def get_car_rviz_markers(car_pos=None, car_yaw=None):
    """Return a list of RViz Marker objects for the car, platform, and inlet."""
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    import tf.transformations as tft

    if car_pos is None:
        car_pos = CAR_POSITION_DEFAULT
    if car_yaw is None:
        car_yaw = CAR_YAW_DEFAULT

    markers = []
    quat = tft.quaternion_from_euler(0, 0, car_yaw)

    def _make(ns, mid, mtype, pos, scale, rgba, orientation=None):
        m = Marker()
        m.header.frame_id = "world"
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        if orientation is not None:
            m.pose.orientation.x = orientation[0]
            m.pose.orientation.y = orientation[1]
            m.pose.orientation.z = orientation[2]
            m.pose.orientation.w = orientation[3]
        else:
            m.pose.orientation.w = 1.0
        m.scale.x = scale[0]
        m.scale.y = scale[1]
        m.scale.z = scale[2]
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    # Platform
    markers.append(_make("car", 10, Marker.CUBE,
                         [car_pos[0], car_pos[1], PLATFORM_HEIGHT / 2],
                         list(PLATFORM_SIZE), (0.5, 0.5, 0.5, 0.8)))

    # Car body
    markers.append(_make("car", 11, Marker.CUBE,
                         car_pos.tolist(),
                         list(CAR_BODY_SIZE), (0.6, 0.6, 0.65, 0.9),
                         orientation=quat))

    # Inlet face (green square)
    inlet_xyz, inlet_R = get_inlet_pose(car_pos, car_yaw)
    markers.append(_make("car", 12, Marker.CUBE,
                         inlet_xyz.tolist(),
                         list(INLET_SIZE), (0.0, 0.9, 0.0, 1.0),
                         orientation=quat))

    # Inlet normal arrow
    m_arrow = _make("car", 13, Marker.ARROW,
                    inlet_xyz.tolist(),
                    [0.10, 0.015, 0.015], (1.0, 0.2, 0.2, 0.9))
    # Point arrow along approach direction
    approach_dir = inlet_R[:, 0]
    tip = inlet_xyz + 0.10 * approach_dir
    m_arrow.points = [Point(x=inlet_xyz[0], y=inlet_xyz[1], z=inlet_xyz[2]),
                      Point(x=tip[0], y=tip[1], z=tip[2])]
    m_arrow.scale.x = 0.012  # shaft diameter
    m_arrow.scale.y = 0.025  # head diameter
    m_arrow.scale.z = 0.0
    markers.append(m_arrow)

    return markers
