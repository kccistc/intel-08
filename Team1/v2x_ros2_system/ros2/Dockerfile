# 1. 기반 이미지 선택
FROM ros:foxy-ros-base

# 2. apt 패키지 설치 및 rosdep 설정 (한 번의 RUN으로 통합)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    git \
    python3-rosdep \
    python3-pip\
    python3-colcon-common-extensions \
    python3-vcstool \
    ros-foxy-rqt \
    ros-foxy-cv-bridge \
    ros-foxy-pcl-conversions \
    ros-foxy-rviz2 \
    ros-foxy-ackermann-msgs \
    ros-foxy-derived-object-msgs \
    ros-foxy-carla-msgs \
    ros-foxy-tf-transformations \
    ros-foxy-rqt-graph && \
    rm -rf /var/lib/apt/lists/*


# 3. ROS 워크스페이스 설정
WORKDIR /ws

# 4. 의존성 설치를 위해 패키지 정의 파일만 먼저 복사
COPY src/car_comms/package.xml src/car_comms/
COPY src/car_control/package.xml src/car_control/
COPY src/car_msgs/package.xml src/car_msgs/
COPY src/car_planning/package.xml src/car_planning/

# 5. carla-ros-bridge 클론 및 전체 의존성 설치
RUN git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git src/ros-bridge && \
    . /opt/ros/foxy/setup.sh && \
    rosdep init || echo "rosdep already initialized" && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --rosdistro foxy

# 6. 전체 소스 코드 복사
COPY . .

# 7. 워크스페이스 빌드
RUN . /opt/ros/foxy/setup.sh && colcon build --symlink-install

# .bashrc에 ROS 환경 설정 추가
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /ws/install/setup.bash" >> ~/.bashrc &&\
    echo "export PYTHONPATH=\"/ws/carla-0.9.13-py3.7-linux-x86_64.egg:\$PYTHONPATH\"" >> ~/.bashrc

ENV PYTHONPATH="/ws/carla-0.9.13-py3.7-linux-x86_64.egg"

# 8. 컨테이너 실행 시 기본 명령어 설정
# 개발용 bash 셸 실행
# To run the application: docker run ros2_v2x:latest bash -c "source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 launch car_planning v2x_full_stack.launch.py enable_serial:=false"
CMD ["bash"]
