import re

with open('temp.sdf', 'r') as f:
    temp_content = f.read()

# Extract models
def extract_model(name):
    # Match <model name='name'> ... </model>
    pattern = r'(<model name=[\'"]' + name + r'[\'"]>.*?</model>)'
    match = re.search(pattern, temp_content, re.DOTALL)
    if match:
        # replace single quotes with double quotes for consistency if desired
        return match.group(1).replace("'", '"')
    return None

inlet = extract_model('hollow_refuel_inlet')
l_wall = extract_model('l_wall')
pillar = extract_model('dynamic_pillar')

with open('kuka_refuel_ws/src/kuka_kr6_gazebo/worlds/refuel_world.sdf', 'r') as f:
    real_sdf = f.read()

# Replace models in real_sdf
def replace_model(name, new_xml):
    global real_sdf
    pattern = r'<model name=[\'"]' + name + r'[\'"]>.*?</model>'
    real_sdf = re.sub(pattern, new_xml, real_sdf, flags=re.DOTALL)

replace_model('hollow_refuel_inlet', inlet)
replace_model('l_wall', l_wall)
replace_model('dynamic_pillar', pillar)

with open('kuka_refuel_ws/src/kuka_kr6_gazebo/worlds/refuel_world.sdf', 'w') as f:
    f.write(real_sdf)

print("Updated refuel_world.sdf with models from temp.sdf")
