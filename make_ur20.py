import os
import xml.etree.ElementTree as ET

MODEL_DIR = "kuka_refuel_ws/src/kuka_kr6_gazebo/models/ur20"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "model.config"), "w") as f:
    f.write("""<?xml version="1.0"?>
<model>
  <name>ur20</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Universal Robots UR20 Fixed</description>
</model>
""")

print("Generating UR20 URDF...")
# Call ur_gz.urdf.xacro directly — it already defines the world link
# internally, so no wrapper needed (a wrapper with <link name="world"/>
# would create a duplicate and crash gz sdf -p).
ret = os.system(
    "xacro $(ros2 pkg prefix --share ur_simulation_gz)/urdf/ur_gz.urdf.xacro"
    " name:=ur ur_type:=ur20 safety_limits:=true > ur20.urdf"
)
if ret != 0:
    raise RuntimeError("xacro failed — is ur_simulation_gz installed?")

print("Converting to SDF...")
ret = os.system("gz sdf -p ur20.urdf > ur20_raw.sdf")
if ret != 0:
    raise RuntimeError("gz sdf -p failed")

print("Patching SDF...")
tree = ET.parse("ur20_raw.sdf")
root = tree.getroot()
model = root.find('model')

# ── 1. Remove gz_ros2_control (crashes without full ROS stack) ────────────
for p in list(model.findall("plugin")):
    if "gz_ros2_control" in p.attrib.get('filename', ''):
        model.remove(p)

# ── 2. Deduplicate world→base_link joints ─────────────────────────────────
# gz sdf -p and our wrapper can both emit a fixed joint from world to
# base_link, causing "[WeldJoint] already has a parent joint" error.
# Keep exactly one.
world_joints = [
    j for j in model.findall("joint")
    if (j.find("parent") is not None and j.find("child") is not None
        and j.find("parent").text in ("world", "world::world")
        and j.find("child").text in ("base_link", "ur20::base_link"))
]
for j in world_joints[1:]:      # remove every duplicate after the first
    model.remove(j)
    print(f"  Removed duplicate world joint: {j.attrib.get('name')}")

if world_joints:
    print(f"  Kept world joint: {world_joints[0].attrib.get('name')}")
else:
    # No world joint found at all — add one explicitly
    wj = ET.SubElement(model, "joint", name="world_joint", type="fixed")
    ET.SubElement(wj, "parent").text = "world"
    ET.SubElement(wj, "child").text = "base_link"
    print("  Added missing world_joint")

# ── 3. Inject JointPositionController plugins ─────────────────────────────
joints = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]
for j in joints:
    plugin = ET.Element(
        "plugin",
        filename="gz-sim-joint-position-controller-system",
        name="gz::sim::systems::JointPositionController",
    )
    ET.SubElement(plugin, "joint_name").text = j
    ET.SubElement(plugin, "initial_position").text = "0.0"
    ET.SubElement(plugin, "p_gain").text  = "500000.0"
    ET.SubElement(plugin, "i_gain").text  = "500.0"
    ET.SubElement(plugin, "d_gain").text  = "20000.0"
    ET.SubElement(plugin, "i_max").text   = "10000.0"
    ET.SubElement(plugin, "i_min").text   = "-10000.0"
    ET.SubElement(plugin, "cmd_max").text = "1000000.0"
    ET.SubElement(plugin, "cmd_min").text = "-1000000.0"
    model.append(plugin)

tree.write(os.path.join(MODEL_DIR, "model.sdf"))
print("Done — ur20 model saved. Run: python3 make_ur20.py && gz sim -r worlds/refuel_gas_station.sdf")
