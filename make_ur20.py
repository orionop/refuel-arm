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
    'shoulder_lift_joint':  0.0,
    'elbow_joint':          0.0,
    'wrist_1_joint':        0.0,
    'wrist_2_joint':        0.0,
    'wrist_3_joint':        0.0,
}
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "model.config"), "w") as f:
    f.write("""<?xml version="1.0"?>
<model>
  <name>ur20</name>
  <version>1.0</version>
  <sdf version="1.10">model.sdf</sdf>
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

# ── 2. Strip the internal "world" link and rewire the world joint ─────────────
# gz sdf -p converts the URDF <link name="world"/> into an actual SDF link
# inside the model. That link is NOT the Gazebo world frame — it's just a
# floating rigid body, so the whole arm falls from its base when physics runs.
# Fix: remove the internal world link and all world→base_link joints, clear any
# relative_to references that pointed at them, then add one clean world_joint
# (now "world" in the joint refers to the real Gazebo world frame).

# Collect the internal world link names that gz sdf -p creates
internal_world_links = [
    l.attrib.get('name', '') for l in model.findall('link')
    if l.attrib.get('name', '') in ('world', 'ur::world', 'base')
       and l.find('collision') is None and l.find('visual') is None
]
for lname in internal_world_links:
    for l in list(model.findall('link')):
        if l.attrib.get('name', '') == lname:
            model.remove(l)
            print(f"  Removed internal world link: {lname}")

# Remove all fixed joints whose parent is the (now-removed) world link or "world"
removed_joint_names = set()
for j in list(model.findall('joint')):
    p_el = j.find('parent')
    c_el = j.find('child')
    if p_el is None or c_el is None:
        continue
    parent_name = p_el.text or ''
    if parent_name in ('world', 'ur::world') or parent_name in internal_world_links:
        removed_joint_names.add(j.attrib.get('name', ''))
        model.remove(j)
        print(f"  Removed old world joint: {j.attrib.get('name')}")

# Clear any link pose that had relative_to pointing at the removed joints
for link in model.findall('link'):
    pose = link.find('pose')
    if pose is not None and pose.get('relative_to', '') in removed_joint_names:
        del pose.attrib['relative_to']
        print(f"  Cleared dangling relative_to on: {link.attrib.get('name')}")

if not FREE_MODE:
    # Add one authoritative fixed joint that welds base_link to the real world frame
    wj = ET.SubElement(model, 'joint', name='world_joint', type='fixed')
    ET.SubElement(wj, 'parent').text = 'world'
    ET.SubElement(wj, 'child').text  = 'base_link'
    print("  Added world_joint (world → base_link)")
else:
    print("  FREE MODE — no world_joint, arm is draggable (pause physics first)")

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

    dynamics = axis.find('dynamics')
    if dynamics is None:
        dynamics = ET.SubElement(axis, 'dynamics')
    for tag, val in [('damping', damping_val), ('friction', friction_val)]:
        el = dynamics.find(tag)
        if el is None:
            ET.SubElement(dynamics, tag).text = val
        else:
            el.text = val

# ── 4. Bake upright spawn angles into joint axis <initial_position> ──────────
# SDF 1.10 (Harmonic) supports <axis><initial_position> — this tells the
# physics engine to start the joint at the given angle, so the PID sees
# near-zero error at t=0 and the arm doesn't violently snap on Play.

joint_map = {j.attrib.get('name', ''): j for j in model.findall('joint')}

for jname, angle in UPRIGHT_JOINTS.items():
    if abs(angle) < 1e-9:
        continue
    j_el = joint_map.get(jname)
    if j_el is None:
        continue
    axis_el = j_el.find('axis')
    if axis_el is None:
        continue
    ip = axis_el.find('initial_position')
    if ip is None:
        ip = ET.SubElement(axis_el, 'initial_position')
    ip.text = f"{angle:.10f}"
    print(f"  initial_position set: {jname} = {angle:.4f} rad")

# ── 5. Inject JointPositionController plugins (skipped in --free mode) ───────
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
        ET.SubElement(plugin, "d_gain").text  = "50000.0"
        ET.SubElement(plugin, "i_max").text   = "500.0"
        ET.SubElement(plugin, "i_min").text   = "-500.0"
        ET.SubElement(plugin, "cmd_max").text = "1000.0"
        ET.SubElement(plugin, "cmd_min").text = "-1000.0"
        model.append(plugin)

tree.write(os.path.join(MODEL_DIR, "model.sdf"))
print("Done — ur20 model saved. Run: python3 make_ur20.py && gz sim -r worlds/refuel_gas_station.sdf")
