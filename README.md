# CORE: Training & Simulation Stack #

The environment uses docker compose and ros2_control.

## Services 
Docker compose brings up the following services:
- Redis, on localhost:6379
- MLFlow, on localhost:5000
- env, custom container packaged with ros2_control and mujoco.

## Commands and Usage
First install the necessary VSCode extensions: 
```
bash /scripts/vscode.setup.bash
```
Then bring-up of the whole stack:
```
xhost +local:docker
cd docker
docker compose up
```
The containers won't stop unless explicitly told so.
To bring the stack down:
```
docker compose down
```

The stack is currently formed by three containers:

- env: containing ros2_control and the mujoco simulation environment. This setup enables for deterministic behavior by avoiding publish/subscribe paradigms.
- redis: Contains the redis database
- mlflow: Experiment visualization and data analysis

## Connecting to the docker container
To connect to the container press, open the command palette in VSCode (Ctrl+Shift+P) and select "Devcontainers: Attach to Running Container' then wen prompted select "env".