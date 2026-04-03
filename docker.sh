#!/bin/bash
# Docker network management script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/docker" || exit 1

command="$1"
shift

gpu=0

while getopts "g" opt; do
    case $opt in
        g)  gpu=1 ;;
        *)  echo "Usage: $0 {start|stop|build} [-g] [packages...]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ "$command" == "start" ]; then
    if [ $gpu -eq 1 ]; then
        echo "======================================================================"
        echo "Starting CORE with GPU acceleration"
        echo "======================================================================"
        docker compose -f docker-compose.yaml --profile gpu up
    else
        echo "======================================================================"
        echo "Starting CORE"
        echo "======================================================================"
        docker compose -f docker-compose.yaml up
    fi
elif [ "$command" == "stop" ]; then
    echo "======================================================================"
    echo "Stopping CORE"
    echo "======================================================================"
    docker compose down
elif [ "$command" == "build" ]; then
    if [ $# -gt 0 ]; then
        pkgs="$*"
        echo "======================================================================"
        echo "Building packages: $pkgs"
        echo "======================================================================"
        docker exec -it env bash -c \
            "source /opt/ros/jazzy/setup.bash && \
             source /ros2_ws/install/setup.bash 2>/dev/null; \
             cd /ros2_ws && colcon build --symlink-install --packages-up-to $pkgs && \
             source /ros2_ws/install/setup.bash"
    else
        echo "======================================================================"
        echo "Building all packages"
        echo "======================================================================"
        docker exec -it env bash -c \
            "source /opt/ros/jazzy/setup.bash && \
             source /ros2_ws/install/setup.bash 2>/dev/null; \
             cd /ros2_ws && colcon build --symlink-install && \
             source /ros2_ws/install/setup.bash"
    fi
else
    echo "Usage: $0 {start|stop|build} [-g] [packages...]"
    echo ""
    echo "Commands:"
    echo "  start          Start all services"
    echo "  stop           Stop all services"
    echo "  build          Build all ROS2 packages in the container"
    echo "  build <pkgs>   Build specific packages (and their dependencies)"
    echo ""
    echo "Options:"
    echo "  -g             Enable GPU acceleration (start only)"
    echo ""
    echo "Examples:"
    echo "  $0 start                        # Start without GPU"
    echo "  $0 start -g                     # Start with GPU"
    echo "  $0 build                        # Build all packages"
    echo "  $0 build parol6_launch          # Build parol6_launch and deps"
    echo "  $0 build parol6_launch core_rl  # Build multiple packages"
    exit 1
fi
