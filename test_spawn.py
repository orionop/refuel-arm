
import os, sys, xml.etree.ElementTree as ET
urdf_path = "kuka_refuel_ws/src/kuka_kr6_gazebo/urdf/kr6_r700_2_clean.urdf"
tree = ET.parse(urdf_path)
root = tree.getroot()
print("URDF root:", root.tag)
