import os, subprocess, xml.etree.ElementTree as ET

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

tree.write("ur20_fixed.sdf")
print("Saved ur20_fixed.sdf! Ready to spawn in ROS 2.")
