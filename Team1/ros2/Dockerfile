FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY . .

RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

CMD ["bash", "-lc", "source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch car_planning v2x_full_stack.launch.py enable_serial:=false"]
