import os
import sys
import xml.etree.ElementTree as ET
import math

FREE_MODE = "--free" in sys.argv  # python3 make_ur20.py --free  → no PIDs, arm moves freely

MODEL_DIR = "kuka_refuel_ws/src/kuka_kr6_gazebo/models/ur20"

# UR20 upright "candle" home pose — mirrors how kr6_r700 model.sdf bakes
# link poses so the arm physically spawns upright, not drooped.
# shoulder_lift=-90°, elbow=+90°, wrist_1=-90°, wrist_2=-90°
UPRIGHT_JOINTS = {
    'shoulder_pan_joint':   0.0,
    'shoulder_lift_joint': -math.pi / 2,
    'elbow_joint':          math.pi / 2,
    'wrist_1_joint':       -math.pi / 2,
    'wrist_2_joint':       -math.pi / 2,
    'wrist_3_joint':        0.0,
}
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

# ── 2. Handle world→base_link fixed joints ───────────────────────────────────
world_joints = [
    j for j in model.findall("joint")
    if (j.find("parent") is not None and j.find("child") is not None
        and j.find("parent").text in ("world", "world::world")
        and j.find("child").text in ("base_link", "ur20::base_link"))
]

if FREE_MODE:
    # Remove ALL world joints so the arm is a free body — translate/rotate
    # tool in Gazebo can then drag it.  Pause physics before dragging.
    for j in world_joints:
        model.remove(j)
        print(f"  FREE MODE — removed world joint: {j.attrib.get('name')}")
else:
    # Normal mode: keep exactly one world joint, remove duplicates
    for j in world_joints[1:]:
        model.remove(j)
        print(f"  Removed duplicate world joint: {j.attrib.get('name')}")
    if world_joints:
        print(f"  Kept world joint: {world_joints[0].attrib.get('name')}")
    else:
        wj = ET.SubElement(model, "joint", name="world_joint", type="fixed")
        ET.SubElement(wj, "parent").text = "world"
        ET.SubElement(wj, "child").text = "base_link"
        print("  Added missing world_joint")

# ── 3. Set joint friction + bake initial spawn angles ────────────────────────
friction_val = '5.0' if FREE_MODE else '500.0'   # low friction in free mode so arm moves easily
damping_val  = '2.0' if FREE_MODE else '200.0'

for joint in model.findall('joint'):
    if joint.attrib.get('type') not in ('revolute', 'continuous'):
        continue
    jname = joint.attrib.get('name', '')
    axis = joint.find('axis')
    if axis is None:
        continue

    if jname in UPRIGHT_JOINTS:
        ip = axis.find('initial_position')
        if ip is None:
            ip = ET.SubElement(axis, 'initial_position')
        ip.text = f"{UPRIGHT_JOINTS[jname]:.10f}"

    dynamics = axis.find('dynamics')
    if dynamics is None:
        dynamics = ET.SubElement(axis, 'dynamics')
    for tag, val in [('damping', damping_val), ('friction', friction_val)]:
        el = dynamics.find(tag)
        if el is None:
            ET.SubElement(dynamics, tag).text = val
        else:
            el.text = val

# ── 4. Inject JointPositionController plugins (skipped in --free mode) ───────
if FREE_MODE:
    print("  FREE MODE — no PID controllers, arm moves freely")
else:
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
        ET.SubElement(plugin, "initial_position").text = f"{UPRIGHT_JOINTS[j]:.10f}"
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
