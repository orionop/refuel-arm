#!/usr/bin/env bash
# Launch Gazebo with the obstacle_test world.
# Sets GZ_SIM_RESOURCE_PATH so the `model://kr6_r700` URI resolves to obstacle/models.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GZ_SIM_RESOURCE_PATH="${HERE}/models${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"

exec gz sim -v 4 -r "${HERE}/worlds/obstacle_test.sdf"
