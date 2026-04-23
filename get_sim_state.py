import subprocess

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

print("=== Gazebo Simulation State ===")

print("\n--- Model Poses (World Frame) ---")
models_output = run_cmd(["gz", "model", "--list"])
models = []
for line in models_output.split('\n'):
    l = line.strip()
    if l.startswith('- '):
        models.append(l[2:].strip())

if not models:
    print("No models found. Is Gazebo running?")
else:
    for m in models:
        pose_out = run_cmd(["gz", "model", "-m", m, "-p"])
        lines = [l.strip() for l in pose_out.split('\n') if l.strip() and 'pose' not in l]
        print(f"[{m}]")
        for l in lines:
            print(f"    {l}")
        print()

print("--- KR6/KR8 Arm State ---")
# Try to get the pose of the end-effector link to see exactly where the arm ended up
arm_models = [m for m in models if 'kuka' in m.lower() or 'kr' in m.lower()]
for arm in arm_models:
    print(f"\n[Arm Model: {arm}]")
    
    # Get base_link pose
    base_pose = run_cmd(["gz", "model", "-m", arm, "-l", "base_link", "-p"])
    print("  base_link pose:")
    for l in base_pose.split('\n'):
        if l.strip() and 'pose' not in l:
            print(f"    {l.strip()}")
            
    # Get link_6 pose (end effector)
    link6_pose = run_cmd(["gz", "model", "-m", arm, "-l", "link_6", "-p"])
    print("  link_6 (End Effector) pose:")
    for l in link6_pose.split('\n'):
        if l.strip() and 'pose' not in l:
            print(f"    {l.strip()}")

print("\n===============================")
print("Script finished.")
