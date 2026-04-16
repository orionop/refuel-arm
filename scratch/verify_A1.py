import numpy as np

# KUKA KR6 R700 Axis 1 convention: H1 = [0, 0, -1]
# Shoulder offset P2 = [0.025, 0.0907, 0.192]

def rot_down(theta):
    # Rotation about [0, 0, -1]
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def check_A1_offset(A1_deg, target_xy):
    A1_rad = np.radians(A1_deg)
    # The shoulder position after A1 rotation
    # Base to Axis 2 offset
    P2 = np.array([0.025, 0.0907, 0.192])
    p_shoulder = rot_down(A1_rad) @ P2
    
    # Vector from shoulder to target
    vec_st = target_xy - p_shoulder[:2]
    
    print(f"A1: {A1_deg} deg")
    print(f"Shoulder XY: {p_shoulder[:2]}")
    print(f"Target XY: {target_xy}")
    print(f"Vector Shoulder->Target: {vec_st}")
    # Angle of this vector relative to the arm's rotated X axis?
    # Actually, the arm's X axis points along the rotated [1, 0, 0]
    arm_x = rot_down(A1_rad) @ [1, 0, 0]
    # Check if the vector to target is aligned with arm X
    dot = np.dot(vec_st, arm_x[:2]) / (np.linalg.norm(vec_st) * np.linalg.norm(arm_x[:2]))
    print(f"Alignment (cos theta): {dot:.4f}")

check_A1_offset(-35.2, np.array([0.439, 0.400]))
