#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

# Code isn't baked into the image — it's mounted from the GitLab-managed
# checkout at /workspace/src/ros2_policy_inference. Build it fresh on every
# container start so a `git pull` + restart always runs the latest code.
if [ -d /workspace/src/ros2_policy_inference ]; then
    colcon build --symlink-install --packages-select ros2_policy_inference
    source /workspace/install/setup.bash
fi

exec "$@"
