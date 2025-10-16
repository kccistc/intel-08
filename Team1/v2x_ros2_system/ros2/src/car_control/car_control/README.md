🚗 V2X-ROS2 Vehicle Control Bridge
📖 프로젝트 개요

이 프로젝트는 ROS2 기반 차량 통신·제어 시스템(V2X) 의 일환으로,
라즈베리파이(ROS2 노드) 와 STM32(모터 제어 MCU) 간의 실시간 제어 명령 송신을 담당합니다.

ROS2의 /vehicle/cmd 토픽(geometry_msgs/Twist)을 구독하여,
해당 속도/조향 명령을 UART 직렬 통신(GPIO TX/RX)으로 STM32에 전송합니다.
전송 프레임은 "CMD V:%.3f,Y:%.3f[*LRC]" 형태이며, STM32는 이를 파싱하여 차량을 제어합니다.

🧩 시스템 구성
 ┌──────────────┐
 │  ROS2 (Pi)  │
 │  car_control │
 │  ├ bt_cmd_bridge.py ───────────┐ UART (/dev/serial0)
 │  └ v2x_avoidance_node.py (선택)│
 └──────────────┘                 │
                                 ▼
                      ┌────────────────┐
                      │   STM32 (MCU)  │
                      │  UART RX/TX    │
                      │  모터/서보 제어 │
                      └────────────────┘


과거에는 블루투스 모듈(HC-05 등)로 /dev/rfcomm0을 사용했으나,
현재 버전은 GPIO UART 직결(/dev/serial0)로 안정화됨.

⚙️ 하드웨어 연결
항목	라즈베리파이 GPIO	STM32 핀	설명
TX	GPIO14 (핀 8)	UART RX	Pi → STM32
RX	GPIO15 (핀 10)	UART TX	STM32 → Pi
GND	(예: 핀 6)	GND	공통 접지
전압	3.3V TTL	3.3V TTL	(5V 불가)

⚠️ UART 콘솔 비활성화 필수

sudo raspi-config nonint do_serial 2
sudo reboot

🧰 소프트웨어 구성
주요 ROS2 패키지
패키지	기능
car_control	/vehicle/cmd 구독 및 UART 명령 송신 노드 (bt_cmd_bridge.py)
car_comms	V2X JSON → ROS2 구조화 변환 노드
car_msgs	ROS2 커스텀 메시지 정의
car_control_bt (선택)	RFCOMM 소켓 기반 C++ 버전
🔧 설치 및 빌드
# 1. 종속 패키지
sudo apt update
sudo apt install -y python3-serial

# 2. 워크스페이스 빌드
cd /ws
colcon build --merge-install --symlink-install
source install/setup.bash

▶️ 실행
1) UART 브리지 노드 실행
ros2 run car_control bt_cmd_bridge --ros-args \
  -p device:=/dev/serial0 \
  -p baud:=115200 \
  -p write_timeout_s:=1.5 \
  -p use_lrc:=False \
  -p verbose:=True


첫 실행 시 로그:

[INFO] [bt_cmd_bridge]: Subscribing Twist on /vehicle/cmd
[INFO] [bt_cmd_bridge]: UART bridge starting: dev=/dev/serial0 baud=115200 ...
[INFO] [bt_cmd_bridge]: Serial opened

2) 명령 퍼블리시 테스트
ros2 topic pub -r 5 /vehicle/cmd geometry_msgs/Twist "{linear: {x: 0.3}, angular: {z: 0.1}}"


STM32 UART 수신 예시:

CMD V:0.300,Y:0.100

🧩 UART 프레임 구조
필드	설명
CMD	명령 헤더
V	선속도 (m/s)
Y	조향 값 (rad 또는 정규화)
*LRC	(선택) XOR 기반 무결성 체크

예:

CMD V:0.500,Y:-0.200*4F

🕵️ 디버깅 및 점검
명령	설명
ls -l /dev/serial0	UART 장치 경로 확인
stty -F /dev/serial0 115200 -echo -ixon -ixoff -crtscts	수동 설정
ros2 topic info /vehicle/cmd -v	퍼블리셔/구독자 연결 확인
screen /dev/serial0 115200	STM32 수신 로그 직접 보기
journalctl -u ssh -f	SSH 연결 상태 확인
🧠 참고 및 향후 계획

C++ RFCOMM 버전 (bt_cmd_bridge_c.cpp) 개발 중
→ 블루투스 연결 시 /dev/rfcomm0 없이 소켓 직접 통신.

피드백 루프 확장: STM32 → ROS2 상태 보고(/vehicle/status) 노드 추가 예정.

AGL/Jetson 연계: /vehicle/cmd 상위 노드로 V2X 회피 판단(car_control/v2x_avoidance_node.py) 연동 가능.
