
# Systemd Service Templates

이 폴더는 라즈베리파이(혹은 리눅스)에서 **부팅 시 자동 기동**을 위한 systemd 서비스 예시를 제공합니다.
실제 환경에 맞게 아래 플레이스홀더를 바꾼 뒤 설치하세요.

- `<YOUR_USER>`, `<YOUR_GROUP>`: 실행 계정/그룹 (예: pi, pi)
- `<PROJECT_ROOT>`: 프로젝트 루트 (예: /home/pi/ros2)
- `<V2X_DIR>`: V2X 클라이언트 경로 (예: /opt/v2x)
- `<PYTHON_BIN>`: 파이썬 경로 (예: /usr/bin/python3)
- `<DOCKER_BIN>`: 도커 바이너리 경로 (예: /usr/bin/docker)
- `<SERIAL_DEV>`: 시리얼 장치 (예: /dev/ttyUSB0)

## 설치 방법
```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
# 필요한 서비스만 enable/start 하세요:
sudo systemctl enable ros2_v2x.service
sudo systemctl start  ros2_v2x.service
