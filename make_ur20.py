import os
import subprocess
import xml.etree.ElementTree as ET

MODEL_DIR = "kuka_refuel_ws/src/kuka_kr6_gazebo/models/ur20"
os.makedirs(MODEL_DIR, exist_ok=True)

# Create model.config
config = f"""<?xml version="1.0"?>
<model>
  <name>ur20</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>Universal Robots UR20 Fixed</description>
</model>
"""
with open(os.path.join(MODEL_DIR, "model.config"), "w") as f:
    f.write(config)

print("Generating UR20 URDF...")
os.system("xacro /opt/ros/jazzy/share/ur_simulation_gz/urdf/ur_gz.urdf.xacro name:=ur ur_type:=ur20 safety_limits:=true > ur20.urdf")

print("Converting to SDF...")
os.system("gz sdf -p ur20.urdf > ur20_raw.sdf")

print("Patching SDF to bypass gz_ros2_control segfault...")
tree = ET.parse("ur20_raw.sdf")
model = tree.getroot().find('model')

# Remove crashing plugin
for p in model.findall("plugin"):
    if "gz_ros2_control-system" in p.attrib.get('filename', ''):
        model.remove(p)

# Auto-detect root link (a link that is never a child in any joint)
child_links = set(j.find('child').text for j in model.findall('joint') if j.find('child') is not None)
all_links = set(l.attrib.get('name') for l in model.findall('link'))
root_links = list(all_links - child_links)
root_link = root_links[0] if root_links else "base_link"
print(f"Auto-detected root link: {root_link}")

# Inject fixed world joint so it doesn't fall down
world_joint = ET.Element("joint", name="world_joint", type="fixed")
ET.SubElement(world_joint, "parent").text = "world"
ET.SubElement(world_joint, "child").text = root_link
model.append(world_joint)

# Inject native PID controllers
joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


for j in joints:
    plugin = ET.Element("plugin", filename="gz-sim-joint-position-controller-system", name="gz::sim::systems::JointPositionController")
    ET.SubElement(plugin, "joint_name").text = j
    ET.SubElement(plugin, "p_gain").text = "20000.0"
    ET.SubElement(plugin, "i_gain").text = "100.0"
    ET.SubElement(plugin, "d_gain").text = "500.0"
    ET.SubElement(plugin, "cmd_max").text = "1000.0"
    ET.SubElement(plugin, "cmd_min").text = "-1000.0"
    model.append(plugin)

tree.write(os.path.join(MODEL_DIR, "model.sdf"))
print("Saved ur20 model into Gas Station models directory! Ready to spawn in ROS 2.")
