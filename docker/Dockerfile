FROM osrf/ros:jazzy-desktop-full-noble


# Python + rendering dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libgl1-mesa-dri libglew-dev libglfw3-dev \
    python3-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment
RUN python3 -m venv core
ENV PATH="/core/bin:$PATH"
RUN pip install mujoco empy lark numpy catkin_pkg stable-baselines3 gymnasium mlflow redis

# ros2_control framework
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-hardware-interface \
    ros-jazzy-joint-state-broadcaster \
    ros-jazzy-joint-trajectory-controller \
    ros-jazzy-ros2-control-cmake \
    && rm -rf /var/lib/apt/lists/*

# ros2_control mujoco binding, caveat: main tracks the latest ROS release, it will break in th future.
RUN mkdir -p /ros2_ws/core/thirdparty && \
    cd /ros2_ws/core/thirdparty && \
    git clone https://github.com/ros-controls/mujoco_ros2_control.git -b main && \  
    git clone https://github.com/pal-robotics/mujoco_vendor.git -b master && \
    cd /ros2_ws && \
    . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install --packages-up-to mujoco_ros2_control

# Parol6 urdfs + meshes only sparse checkout (temporary solution)
RUN mkdir -p /ros2_ws/core/thirdparty/parol6_upstream && \
    cd /ros2_ws/core/thirdparty/parol6_upstream && \
    git init && \
    git remote add origin https://github.com/PCrnjak/PAROL6-Desktop-robot-arm.git && \
    git config core.sparseCheckout true && \
    echo "PAROL6_URDF/" >> .git/info/sparse-checkout && \
    git pull --depth 1 origin main

# Utils
RUN apt-get update && apt-get install -y \
    iputils-ping \
    redis-tools \
    libabsl-dev \
    ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws/core/src

# Setup bashrc for interactive shells
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash' >> /root/.bashrc && \
    echo 'export PATH="/core/bin:$PATH"' >> /root/.bashrc && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]
CMD ["bash"]