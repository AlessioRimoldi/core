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
        *)  echo "Usage: $0 {start|stop|build|build-packages} [-g] [packages...]"; exit 1 ;;
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
        docker compose -f docker-compose.yaml --profile cpu up
    fi
elif [ "$command" == "stop" ]; then
    echo "======================================================================"
    echo "Stopping CORE"
    echo "======================================================================"
    docker compose --profile cpu --profile gpu down
elif [ "$command" == "build" ]; then
    echo "======================================================================"
    echo "Rebuilding containers"
    echo "======================================================================"
    docker compose -f docker-compose.yaml --profile cpu --profile gpu build
elif [ "$command" == "build-packages" ]; then
    # Detect which env container is running (env-cpu or env-gpu)
    container=""
    for candidate in env-gpu env-cpu; do
        if docker ps --format '{{.Names}}' | grep -q "^${candidate}$"; then
            container="$candidate"
            break
        fi
    done
    if [ -z "$container" ]; then
        echo "Error: no env container running. Start one with './docker.sh start' or './docker.sh start -g'." >&2
        exit 1
    fi
    echo "Using container: $container"

    if [ $# -gt 0 ]; then
        pkgs="$*"
        echo "======================================================================"
        echo "Building packages: $pkgs"
        echo "======================================================================"
        docker exec -it "$container" bash -c \
            "source /opt/ros/jazzy/setup.bash && \
             source /ros2_ws/install/setup.bash 2>/dev/null; \
             cd /ros2_ws && colcon build --symlink-install --packages-up-to $pkgs && \
             source /ros2_ws/install/setup.bash"
    else
        echo "======================================================================"
        echo "Building all packages"
        echo "======================================================================"
        docker exec -it "$container" bash -c \
            "source /opt/ros/jazzy/setup.bash && \
             source /ros2_ws/install/setup.bash 2>/dev/null; \
             cd /ros2_ws && colcon build --symlink-install && \
             source /ros2_ws/install/setup.bash"
    fi
else
    echo "Usage: $0 {start|stop|build|build-packages} [-g] [packages...]"
    echo ""
    echo "Commands:"
    echo "  start                  Start all services"
    echo "  stop                   Stop all services"
    echo "  build                  Rebuild Docker containers"
    echo "  build-packages         Build all ROS2 packages in the container"
    echo "  build-packages <pkgs>  Build specific packages (and their dependencies)"
    echo ""
    echo "Options:"
    echo "  -g             Enable GPU acceleration (start only)"
    echo ""
    echo "Examples:"
    echo "  $0 start                                # Start without GPU"
    echo "  $0 start -g                             # Start with GPU"
    echo "  $0 build                                # Rebuild containers"
    echo "  $0 build-packages                       # Build all packages"
    echo "  $0 build-packages parol6_launch          # Build parol6_launch and deps"
    echo "  $0 build-packages parol6_launch core_rl  # Build multiple packages"
    exit 1
fi
