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

# ── 4. Bake upright spawn angles via child-link pose (relative_to joint frame) ─
# <axis>/<initial_position> is not in this system's sdformat schema so it is
# silently ignored. The correct SDF approach: set each child link's
# <pose relative_to="joint_name"> to encode the desired initial angle.
# For a Y-axis revolute at angle θ: pose = "0 0 0 0 θ 0"
# For a Z-axis revolute at angle θ: pose = "0 0 0 0 0 θ"
# (UR link origins coincide with their joint frame, so translation is 0.)

joint_map = {j.attrib.get('name', ''): j for j in model.findall('joint')}
link_map   = {l.attrib.get('name', ''): l for l in model.findall('link')}

for jname, angle in UPRIGHT_JOINTS.items():
    if abs(angle) < 1e-9:
        continue
    j_el = joint_map.get(jname)
    if j_el is None:
        continue
    child_name = (j_el.find('child').text or '').strip()
    child_link = link_map.get(child_name)
    if child_link is None:
        print(f"  WARNING: link '{child_name}' not found for {jname}")
        continue

    # Determine dominant axis from the joint definition
    axis_el = j_el.find('axis')
    xyz_el  = axis_el.find('xyz') if axis_el is not None else None
    xyz_str = (xyz_el.text or '0 0 1').strip() if xyz_el is not None else '0 0 1'
    vals    = [float(v) for v in xyz_str.split()]
    if abs(vals[1]) > 0.5:          # Y-dominant → pitch
        pose_text = f"0 0 0 0 {angle:.10f} 0"
    elif abs(vals[2]) > 0.5:        # Z-dominant → yaw
        pose_text = f"0 0 0 0 0 {angle:.10f}"
    else:                            # X-dominant → roll
        pose_text = f"0 0 0 {angle:.10f} 0 0"

    pose_el = child_link.find('pose')
    if pose_el is None:
        pose_el = ET.SubElement(child_link, 'pose')
    pose_el.attrib['relative_to'] = jname
    pose_el.text = pose_text
    print(f"  Pose baked: {child_name} relative_to {jname} = [{pose_text}]")

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
