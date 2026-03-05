# CORE: Training & Simulation Stack #

The environment uses docker compose and ros2_control.

## Services 
Docker compose brings up the following services:
- Redis, on localhost:6379
- MLFlow, on localhost:5000

## Commands and Usage
From the project root folder. Bring-up of the whole stack:S
```
xhost +local:docker
cd docker
docker compose up
```