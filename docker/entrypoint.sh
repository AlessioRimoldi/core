#!/bin/bash
# Docker env entrypoint script
set -e

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Source third-party workspace (mujoco_ros2_control etc.) built during image build
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Build user packages from the bind-mounted src/ directory
cd /ros2_ws

# Clean stale ament_cmake_python directories that conflict with --symlink-install
# Only removes dirs where dirname == package name (the specific symlink conflict)
for d in /ros2_ws/build/*/ament_cmake_python/*/; do
    pkg=$(basename "$(dirname "$d")")
    target="$d$pkg"
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "==> Cleaning stale dir: $target"
        rm -rf "$target"
    fi
done

if [ -n "$ROBOT_PKG" ]; then
    echo "==> Building packages up to: $ROBOT_PKG"
    colcon build --symlink-install --packages-up-to $ROBOT_PKG || {
        echo "==> WARNING: Build failed. Starting shell anyway for debugging."
    }
else
    echo "==> Building all packages in src/"
    colcon build --symlink-install || {
        echo "==> WARNING: Build failed. Starting shell anyway for debugging."
    }
fi

# Re-source to pick up newly built packages
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Execute passed command
exec "$@"