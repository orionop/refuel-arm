# Obstacle Workspace (KUKA KR6 R700)

Reactive obstacle-avoidance safety system for the KUKA KR6 R700 industrial manipulator. This module provides high-rigor implementations of state-of-the-art safety algorithms, benchmarked for deterministic performance before deployment to ROS 1 Noetic / Gazebo.

## 1. System Stack

- **Core Logic:** Pure Python 3.9+ (Zero-dependency safety methods)
- **Robotics Integration:** ROS 1 Noetic (Ubuntu 20.04)
- **Simulation:** Gazebo 11
- **Arm:** KUKA KR6 R700 (6-DOF, 0.70m reach)
- **Optimization:** `qpsolvers` with OSQP backend for real-time safety filtering.

## 2. Directory Structure

```
obstacle/
├── safety/                          # Core Algorithmic Library
│   ├── types.py                     # Common data structures (RobotState, Obstacle)
│   ├── kinematics.py                # KR6 R700 PoE Kinematics (Forward + Jacobian)
│   ├── methods/                     # Paper Implementations
│   │   ├── base.py                  # Abstract SafetyMethod interface
│   │   ├── threshold.py             # Baseline: Distance Threshold + E-Stop
│   │   ├── apf.py                   # 1: APF + Informed Circular Fields (Becker 2024)
│   │   ├── neo.py                   # 2: NEO Velocity Damper (Haviland 2021)
│   │   ├── hocbf.py                 # 3: HOCBF Safety Filter (Singletary 2022)
│   │   └── _qp.py                   # QP Solver Wrapper (OSQP/Scipy)
│   └── harness/                     # Benchmarking Tools
│       ├── scenarios.py             # 10 Deterministic Test Scenarios
│       ├── metrics.py               # Separation, Jerk, Reaction Time metrics
│       └── runner.py                # Simulation loop for offline verification
├── scripts/
│   └── benchmark.py                 # CLI: Run the 40-test comparison suite
└── README.md
```

## 3. Implemented Methods

| Method | Type | Reference |
| :--- | :--- | :--- |
| **Distance Threshold** | Baseline | Traditional e-stop on safety violation. |
| **APF (Circular)** | Vector Field | Khatib (1985) + Becker (2024) Informed Swirl. |
| **NEO** | Reactive QP | Haviland & Corke (2021) Velocity Damper. |
| **HOCBF** | Safety Filter | Singletary et al. (2022) Barrier Functions. |

## 4. Benchmarking (Offline)

The benchmark evaluates methods on the **KR6 R700** in the **Candle Pose** ($q = [0, -90^\circ, 90^\circ, 0, 0, 0]$).

To run the full suite:
```bash
# Ensure dependencies are installed
pip install qpsolvers osqp numpy scipy

# Run benchmark
python3 obstacle/scripts/benchmark.py
```

## 5. ROS 1 Noetic Integration (Gazebo)

The safety methods are designed to be "plug-and-play" within a ROS 1 node. The `SafetyMethod.step()` function consumes a `RobotState` (derived from `/joint_states`) and `Obstacle` data (derived from YOLOv11/Point Cloud) to produce a safe `qdot_cmd` for the KUKA joint trajectory controllers.

---
**Note:** For the meeting on May 12th, refer to `artifacts/benchmark_report.md` for a detailed analysis of method performance on the KR6 R700 hardware.
