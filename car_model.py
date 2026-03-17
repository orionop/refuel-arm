#!/usr/bin/env python3
"""
Car Model: Gazebo DAE-mesh car spawning and fuel inlet pose computation
========================================================================

Spawns a realistic car from gazebo_cars (DAE mesh) on an elevated platform
in Gazebo and computes the fuel inlet target pose for the KUKA KR6 R700.

Car meshes come from: https://github.com/ayushgaud/gazebo_cars
Requires: gazebo_cars/ cloned inside the refuel-arm repo root.
"""
import sys
import os
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(_THIS_DIR, 'kuka_refuel_ws', 'src',
                                'kuka_kr6_gazebo', 'scripts'))
from ik_geometric import rot

# ── Available car models and their mesh files ─────────────────────
MODELS_DIR = os.path.join(_THIS_DIR, 'gazebo_cars', 'models')

# Mesh filename lookup (most use car_name.dae, some differ)
_MESH_NAMES = {
    'car_golf': 'golf.dae',
    'car_beetle': 'beetle.dae',
    'car_lexus': 'lexus.dae',
    'car_opel': 'opel.dae',
    'car_polo': 'polo.dae',
    'car_volvo': 'volvo.dae',
}

CAR_MODEL_DEFAULT = 'car_golf'
CAR_SCALE_DEFAULT = 0.10           # Real car ~4.2m → scaled ~0.42m

# ── Geometry Constants ────────────────────────────────────────────
PLATFORM_HEIGHT = 0.35
PLATFORM_SIZE   = (0.65, 0.45, PLATFORM_HEIGHT)

# Approximate scaled car envelope (at 0.10x of a typical sedan ~4.2 x 1.8 x 1.4m)
CAR_APPROX_SIZE = (0.42, 0.18, 0.14)

INLET_SIZE = (0.06, 0.005, 0.06)    # visible green square marker

# Car center sits on the platform; Z adjusted for wheel contact
CAR_POSITION_DEFAULT = np.array([0.55, 0.35, PLATFORM_HEIGHT + 0.01])
CAR_YAW_DEFAULT      = 0.0

# Inlet offset in car's local frame (-Y side = toward robot, slightly behind center)
INLET_OFFSET_LOCAL = np.array([0.05, -(CAR_APPROX_SIZE[1] / 2 + 0.002), 0.06])

# EE orientation: tool axis pointing +Y (into the car's -Y face)
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


# Green inlet marker (spawned as separate overlay on the car)
INLET_MARKER_SDF = """<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="fuel_inlet_marker">
    <static>true</static>
    <link name="link">
      <visual name="vis">
        <geometry><box><size>{ix} {iy} {iz}</size></box></geometry>
        <material>
          <ambient>0.0 0.85 0.0 1</ambient>
          <diffuse>0.0 0.90 0.0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>""".format(ix=INLET_SIZE[0], iy=INLET_SIZE[1], iz=INLET_SIZE[2])


def _build_mesh_car_sdf(car_name=CAR_MODEL_DEFAULT, scale=CAR_SCALE_DEFAULT):
    """Build SDF for a gazebo_cars mesh model with file:// URI and custom scale."""
    mesh_file = _MESH_NAMES.get(car_name)
    if mesh_file is None:
        # Fallback: look for any .dae in the meshes/ directory
        meshes_dir = os.path.join(MODELS_DIR, car_name, 'meshes')
        if os.path.isdir(meshes_dir):
            daes = [f for f in os.listdir(meshes_dir) if f.endswith('.dae')]
            if daes:
                mesh_file = daes[0]
    if mesh_file is None:
        raise FileNotFoundError(f"No mesh found for model '{car_name}'")

    mesh_path = os.path.join(MODELS_DIR, car_name, 'meshes', mesh_file)
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    s = scale
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.5">
  <model name="refuel_car">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>file://{mesh_path}</uri>
            <scale>{s} {s} {s}</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <box><size>{CAR_APPROX_SIZE[0]} {CAR_APPROX_SIZE[1]} {CAR_APPROX_SIZE[2]}</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""
    return sdf


# ── Pose Computation ──────────────────────────────────────────────

def get_inlet_pose(car_pos, car_yaw=0.0):
    """Compute world-frame inlet position and EE orientation."""
    R_yaw = rot(np.array([0., 0., 1.]), car_yaw)
    inlet_xyz = car_pos + R_yaw @ INLET_OFFSET_LOCAL
    inlet_R = R_yaw @ R_TOOL_INTO_CAR
    return inlet_xyz, inlet_R


def get_preapproach_pose(inlet_xyz, inlet_R, standoff=0.08):
    """Pull the target back from the inlet along the approach normal."""
    approach_dir = inlet_R[:, 0]
    preapproach_xyz = inlet_xyz - standoff * approach_dir
    return preapproach_xyz, inlet_R


# ── Gazebo Spawning ───────────────────────────────────────────────

def spawn_car_gazebo(car_pos=None, car_yaw=None, car_name=CAR_MODEL_DEFAULT,
                     scale=CAR_SCALE_DEFAULT):
    """Spawn platform + mesh car + inlet marker in Gazebo."""
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

    quat = tft.quaternion_from_euler(0, 0, car_yaw)

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

    # 2. Mesh car on platform
    car_sdf = _build_mesh_car_sdf(car_name, scale)
    p_car = Pose()
    p_car.position.x = car_pos[0]
    p_car.position.y = car_pos[1]
    p_car.position.z = car_pos[2]
    p_car.orientation.x = quat[0]
    p_car.orientation.y = quat[1]
    p_car.orientation.z = quat[2]
    p_car.orientation.w = quat[3]
    try:
        spawn("refuel_car", car_sdf, "/", p_car, "world")
    except Exception as e:
        print(f"  Car spawn note: {e}")

    # 3. Green inlet marker (separate overlay)
    inlet_xyz, _ = get_inlet_pose(car_pos, car_yaw)
    p_inlet = Pose()
    p_inlet.position.x = inlet_xyz[0]
    p_inlet.position.y = inlet_xyz[1]
    p_inlet.position.z = inlet_xyz[2]
    p_inlet.orientation.x = quat[0]
    p_inlet.orientation.y = quat[1]
    p_inlet.orientation.z = quat[2]
    p_inlet.orientation.w = quat[3]
    try:
        spawn("fuel_inlet_marker", INLET_MARKER_SDF, "/", p_inlet, "world")
    except Exception as e:
        print(f"  Inlet marker spawn note: {e}")

    print(f"  Spawned platform + {car_name} (scale {scale}) + inlet marker")


# ── RViz Markers ──────────────────────────────────────────────────

def get_car_rviz_markers(car_pos=None, car_yaw=None):
    """Return RViz Marker objects for the car, platform, inlet, and arrow."""
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

    # Car body (RViz uses a box approximation since DAE mesh isn't easy to load)
    markers.append(_make("car", 11, Marker.CUBE,
                         [car_pos[0], car_pos[1], car_pos[2] + CAR_APPROX_SIZE[2] / 2],
                         list(CAR_APPROX_SIZE), (0.6, 0.6, 0.65, 0.7),
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
    approach_dir = inlet_R[:, 0]
    tip = inlet_xyz + 0.10 * approach_dir
    m_arrow.points = [Point(x=inlet_xyz[0], y=inlet_xyz[1], z=inlet_xyz[2]),
                      Point(x=tip[0], y=tip[1], z=tip[2])]
    m_arrow.scale.x = 0.012
    m_arrow.scale.y = 0.025
    m_arrow.scale.z = 0.0
    markers.append(m_arrow)

    return markers
