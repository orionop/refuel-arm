# Obstacle Workspace

Dynamic-obstacle safety system for the KR6 R700, separate from the autonomous
refueling pipeline in `../refueling/`.

## Phase 0 — arm + Gazebo smoke test

```
obstacle/
├── models/kr6_r700/        # KUKA KR6 R700 SDF + meshes (copied from refueling/)
├── worlds/obstacle_test.sdf # ground plane + arm, no obstacles yet
├── scripts/
│   ├── joint_control.py     # gz topic publisher helpers
│   └── test_arm.py          # joint sweep smoke test
└── launch.sh                # gz sim launcher
```

### Run

Terminal A — Gazebo:
```bash
./obstacle/launch.sh
```

Terminal B — joint sweep:
```bash
python3 obstacle/scripts/test_arm.py
```

You should see the arm settle at `J2 = -π/2`, then sweep each joint one at a
time, then return home.

### Notes

- `kuka_robot_descriptions/` (cloned from kroshu) is gitignored. We copied the
  proven `kr6_r700` model from `refueling/kuka_refuel_ws/` for now; the kroshu
  xacros will be revisited when we need the full URDF for kinematics.
- The arm uses `JointPositionController` plugins (one per joint); each listens
  on `/model/kr6_r700/joint/joint_N/0/cmd_pos` (`gz.msgs.Double`).
