#!/bin/bash
# Docker network management script

SCRIPT_DIR="$(cd $(dirname "$0") && pwd)"
cd "$SCRIPT_DIR/docker"

command="$1"
shift 

gpu=0 

while getopts "g" opt; do
    case $opt in 
        g)  gpu=1 ;;
        *)  echo "Usage: $0 [-g]"; exit 1 ;;
    esac
done

if [ "$command" == "start" ]; then
    if [ $gpu -eq 1 ]; then
        echo "======================================================================"
        echo "Starting CORE with GPU accelleration"
        echo "======================================================================"
        docker compose -f docker-compose.yaml -profile gpu up 
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
else 
    echo "Usage: $0 {start|stop} [-g]"
    exit 1
fi