📡 System Overview
[Server] → UDP Multicast(JSON)
    ↓
[Raspberry Pi: V2X Gateway]
    client.py : UDP 수신, HMAC 검증, 중복 제거, TTL 처리
    → /v2x/alert (std_msgs/String)
    ↓
[ROS2 Bridge: car_comms]
    v2x_alert_bridge.py : JSON → V2VAlert(msg)
    /v2x/alert_struct (car_msgs/msg/V2VAlert)
    ↓
[Decision Node: car_planning]
    decision_maker.py : /v2x/alert_struct → /vehicle/cmd (Twist)
    ↓
[STM32 (micro-ROS)]
    /vehicle/cmd 수신 → 모터/서보 제어

🧩 주요 패키지 구성
1️⃣ car_comms

노드: v2x_alert_bridge

입력: /v2x/alert (std_msgs/msg/String)

출력: /v2x/alert_struct (car_msgs/msg/V2VAlert)

기능:

JSON 메시지를 구조화된 ROS2 메시지로 변환

TTL 만료/중복 메시지 필터링

QoS 및 주제명 파라미터 설정 가능

2️⃣ car_msgs

메시지 정의:

V2VAlert.msg

uint32 ver
string src
uint32 seq
builtin_interfaces/Time ts
string type
string severity
float32 distance_m
string road
float64 lat
float64 lon
string suggest
float32 ttl_s


용도: V2X 이벤트 구조체 전달

3️⃣ car_planning

노드: decision_maker

입력: /v2x/alert_struct

출력: /vehicle/cmd (geometry_msgs/msg/Twist)

기능:

이벤트 종류(type), 심각도(severity), 거리(distance_m), 제안(suggest)에 따라 주행 판단

stop_distance_m, slow_distance_m 파라미터 기반 감속/정지 결정

TTL 초과 메시지 무시

⚙️ 자동 실행 구조 (라즈베리파이 기준)
🐳 Docker 기반 자동 기동

컨테이너 이름: ros2_v2x

이미지: ros:jazzy-ros-base

자동 실행 경로: /home/pi/ros2

컨테이너 명령:

bash -lc "source /opt/ros/jazzy/setup.bash && \
          colcon build --symlink-install || true && \
          source install/setup.bash && \
          ros2 launch car_planning v2x_full_stack.launch.py enable_serial:=false"


systemd 서비스: /etc/systemd/system/ros2_v2x.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a ros2_v2x
ExecStop=/usr/bin/docker stop ros2_v2x


부팅 시:
전원 ON → Docker 데몬 → ros2_v2x 컨테이너 자동 실행 → ROS2 런치 자동 시작

🧠 주요 런치 파일
car_planning/launch/v2x_full_stack.launch.py

브릿지(v2x_alert_bridge) + 디시전메이커(decision_maker) + 선택적 모터제어를 통합 실행

인자:

enable_serial:=true|false
serial_port:=/dev/ttyUSB0
baudrate:=115200
stop_distance_m:=8.0
slow_distance_m:=15.0

🧪 테스트 명령어
V2X 이벤트 수동 발행
ros2 topic pub -1 /v2x/alert_struct car_msgs/msg/V2VAlert \
"{ver:1, src:'sim', seq:100, ts:{sec:0,nanosec:0}, type:'collision', severity:'high', distance_m:5.0, road:'A1', lat:0.0, lon:0.0, suggest:'stop', ttl_s:10.0}"

출력 확인
ros2 topic echo /vehicle/cmd


예상 출력:

[EMERGENCY_STOP] type=collision, sev=high, dist=5.0m → cmd: v=0.00 m/s

🔍 모듈별 로그 예시
[car_comms.v2x_alert_bridge] [INFO] Received alert JSON from /v2x/alert
[car_planning.decision_maker] [INFO] [SLOW_DOWN_ADVICE] type=collision, dist=500.0m → cmd: v=0.30 m/s
[car_planning.decision_maker] [WARN] [EMERGENCY_STOP] type=collision, dist=5.0m → cmd: v=0.00 m/s

🧰 개발/운영 명령어
명령	설명
docker logs -f ros2_v2x	실시간 로그 확인
docker exec -it ros2_v2x bash	컨테이너 내부 진입
ros2 topic list	활성 토픽 확인
ros2 node list	활성 노드 확인
systemctl restart ros2_v2x.service	서비스 재시작
ros2 bag record /v2x/alert_struct /vehicle/cmd	데이터 로깅
🧩 향후 확장 계획

Jetson Nano에서 AI 기반 회피 모델 연동

AGL Cluster와 /status 연동하여 시각화 UI 구현

STM32 micro-ROS 통신 최적화

클라우드 → 차량 OTA 이벤트 전달 자동화
