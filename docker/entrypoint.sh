#!/bin/bash
# Docker env entrypoint script
set -e

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Source workspace if built
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Execute passed command
exec "$@"