# Deprecated & Scrapped Approaches

This folder contains older, prototype, or retired code that is **no longer used** in the main project pipeline. It is kept solely for archival and reference purposes to document the development history.

### Contents:
1. **`/matlab`**: The original MATLAB scripts containing prototype IK formulations and URDF debugging from the early stages of the project.
2. **`/ikflow` & `/kaggle`**: Neural network and learning-based approaches to inverse kinematics. These models were evaluated but ultimately deprecated in favor of the current exact algebraic IK-Geo implementation, which proved significantly more accurate and reliable.
3. **`instructions.md`**: Old standalone setup instructions that have now been integrated directly into the main or setup documentation.
4. **`test_ik_geo_pipeline.py` & `test_ik_local.py` & `test_ik.py`**: Earlier prototype pipeline scripts that have since been superseded by the `test_full_pipeline.py` orchestrator and the `ik_trajectories/` tracking scripts.

**⚠️ Do not use these files for execution.** They are unmaintained. For running the project, please refer to the main repository `SETUP.md` and `README.md`.
