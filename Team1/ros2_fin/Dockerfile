# 1. 기반 이미지 선택
FROM arm64v8/ros:foxy-ros-base

# 2. apt 패키지 설치 및 rosdep 설정 (한 번의 RUN으로 통합)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-pip\
    python3-colcon-common-extensions \
    && pip3 install -U pyserial \
    && rm -rf /var/lib/apt/lists/*


# 3. ROS 워크스페이스 설정
WORKDIR /ws

# 4. 의존성 설치를 위해 패키지 정의 파일만 먼저 복사
COPY car_comms/package.xml car_comms/
COPY car_control/package.xml car_control/
COPY car_msgs/package.xml car_msgs/
COPY car_planning/package.xml car_planning/

# 6. 전체 소스 코드 복사
COPY . .

# 7. 워크스페이스 빌드
RUN . /opt/ros/foxy/setup.sh && colcon build --symlink-install

# .bashrc에 ROS 환경 설정 및 PYTHONPATH 추가
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc && \
    echo "source /ws/install/setup.bash" >> ~/.bashrc && \
    echo "export PYTHONPATH=\"/ws/carla-0.9.13-py3.7-linux-x86_64.egg:\$PYTHONPATH\"" >> ~/.bashrc

# PYTHONPATH 환경변수 설정 (런타임용)
ENV PYTHONPATH="/ws/carla-0.9.13-py3.7-linux-x86_64.egg"

# 8. 컨테이너 실행 시 기본 명령어 설정
# 개발용 bash 셸 실행
# To run the application: docker run ros2_v2x:latest bash -c "source /opt/ros/foxy/setup.bash && source install/setup.bash && ros2 launch car_planning v2x_full_stack.launch.py enable_serial:=false"
CMD ["bash"]
