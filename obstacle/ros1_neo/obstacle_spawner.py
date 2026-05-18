#!/usr/bin/env python
"""
Spawns two spherical obstacles in Gazebo and animates them along a
linear velocity vector at a fixed update rate.

Each spawned model name starts with "obs_" so that neo_ros_node.py
picks them up via the /gazebo/model_states topic.

Assumptions:
    - Gazebo Classic 11 with gazebo_ros services live:
        /gazebo/spawn_sdf_model
        /gazebo/set_model_state
        /gazebo/delete_model

Usage:
    rosrun <pkg> obstacle_spawner.py
    or:
    python3 obstacle_spawner.py
"""

import rospy
from gazebo_msgs.srv import SpawnModel, SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, Vector3


# ----- Scene config (mirror of Swift demo) -----

OBSTACLES = [
    {
        "name": "obs_0",
        "radius": 0.05,
        "start": (0.35, 0.5, 0.5),
        "velocity": (0.0, -0.15, 0.0),
    },
    {
        "name": "obs_1",
        "radius": 0.05,
        "start": (0.1, 0.5, 0.7),
        "velocity": (0.0, -0.15, 0.0),
    },
]

UPDATE_RATE_HZ = 50.0  # how fast we re-set the obstacle pose in Gazebo


def sphere_sdf(radius: float, name: str) -> str:
    """Inline SDF for a sphere with no inertia and unit-ish color."""
    return f"""<?xml version='1.0'?>
<sdf version='1.7'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <inertial>
        <mass>0.001</mass>
        <inertia>
          <ixx>0.00001</ixx><iyy>0.00001</iyy><izz>0.00001</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <visual name='visual'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material>
          <ambient>0.9 0.2 0.2 1</ambient>
          <diffuse>0.9 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <collision name='collision'>
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>
      <gravity>0</gravity>
    </link>
  </model>
</sdf>
"""


def make_pose(xyz):
    p = Pose()
    p.position = Point(*xyz)
    p.orientation = Quaternion(0, 0, 0, 1)
    return p


def make_state(name, xyz, vxyz):
    s = ModelState()
    s.model_name = name
    s.pose = make_pose(xyz)
    s.twist = Twist(linear=Vector3(*vxyz), angular=Vector3(0, 0, 0))
    s.reference_frame = "world"
    return s


def main():
    rospy.init_node("obstacle_spawner", anonymous=False)

    rospy.loginfo("Waiting for Gazebo services...")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/set_model_state")
    rospy.wait_for_service("/gazebo/delete_model")

    spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    delete = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    # Delete any leftover obstacles from a prior run (ignore errors)
    for obs in OBSTACLES:
        try:
            delete(obs["name"])
        except Exception:
            pass

    # Spawn each obstacle
    for obs in OBSTACLES:
        sdf = sphere_sdf(obs["radius"], obs["name"])
        ok = spawn(
            model_name=obs["name"],
            model_xml=sdf,
            robot_namespace="",
            initial_pose=make_pose(obs["start"]),
            reference_frame="world",
        )
        rospy.loginfo("Spawned %s: %s", obs["name"], ok.success)

    # Animate: set ModelState at fixed rate
    positions = {obs["name"]: list(obs["start"]) for obs in OBSTACLES}
    velocities = {obs["name"]: obs["velocity"] for obs in OBSTACLES}

    rate = rospy.Rate(UPDATE_RATE_HZ)
    dt = 1.0 / UPDATE_RATE_HZ
    rospy.loginfo("Animating obstacles at %.1f Hz", UPDATE_RATE_HZ)

    # Clean up on shutdown
    def shutdown_cb():
        for obs in OBSTACLES:
            try:
                delete(obs["name"])
            except Exception:
                pass

    rospy.on_shutdown(shutdown_cb)

    while not rospy.is_shutdown():
        for name in positions:
            v = velocities[name]
            positions[name][0] += v[0] * dt
            positions[name][1] += v[1] * dt
            positions[name][2] += v[2] * dt
            try:
                set_state(make_state(name, positions[name], v))
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "set_model_state failed: %s", exc)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
