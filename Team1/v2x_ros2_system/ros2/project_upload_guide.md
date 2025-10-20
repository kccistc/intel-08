# 🚗 V2X ROS2 System — Upload & Backup Guide

이 문서는 Raspberry Pi 또는 VM 환경에서 개발된 **V2X ROS2 차량 통신 시스템**을  
GitHub로 안전하게 업로드하고 백업하기 위한 절차를 안내합니다.

---

## 🧩 1. 프로젝트 구성 개요

ros2/
├── car_msgs/ # ROS2 메시지 정의 패키지
│ └── msg/V2VAlert.msg
├── car_comms/ # V2X Alert Bridge (JSON → ROS2 구조체 변환)
│ └── v2x_alert_bridge.py
├── car_planning/ # 회피 판단(Decision Maker) 노드
│ ├── decision_maker.py
│ ├── launch/
│ │ ├── v2x_decision_pipeline.launch.py
│ │ └── v2x_full_stack.launch.py
│ └── setup.py
├── systemd/ # 자동 기동용 서비스 템플릿
│ ├── ros2_v2x.service
│ ├── v2x-alert-client.service
│ ├── v2x-alert-client-docker.service
│ ├── v2x-gateway.service
│ └── README.md
├── Dockerfile # (선택) 동일 환경 재현용 Dockerfile
├── .gitignore # 빌드 산출물 제외
└── README.md # 프로젝트 설명

yaml
코드 복사

---

## 🧱 2. GitHub 업로드 전 준비

라즈베리파이에서 다음 폴더만 PC로 복사합니다:

/home/pi/ros2/
│
├── car_msgs/
├── car_comms/
├── car_planning/
├── systemd/
├── README.md
├── .gitignore
└── (Dockerfile 있으면 함께)

markdown
코드 복사

> ⚠️ `build/`, `install/`, `log/` 폴더는 자동 생성되므로 제외하세요.

---

## 💻 3. 폴더 가져오기 (라즈베리파이 → PC)

### 🪟 Windows (WinSCP 사용)
1. WinSCP 실행  
   - 프로토콜: `SFTP`
   - 호스트: Pi IP (예: `192.168.0.10`)
   - 사용자명: `pi`
   - 비밀번호: (라즈베리파이 비밀번호)
2. `/home/pi/ros2` 폴더 열기 → 전체 복사 → PC로 저장

### 💻 macOS / Linux (scp 명령)
```bash
scp -r pi@192.168.x.x:/home/pi/ros2 ~/Downloads/ros2_backup
💡 VSCode Remote SSH
VSCode 설치 + “Remote - SSH” 확장 추가

좌측 하단 초록 버튼 → “Connect to Host…” → pi@192.168.x.x

/home/pi/ros2 열기 → 복사 또는 직접 Push 가능

🚀 4. GitHub 업로드 절차
1️⃣ GitHub에서 새 리포지토리 생성
예: v2x-ros2-system

Public/Private 자유 선택

README, .gitignore 자동생성은 OFF

2️⃣ 로컬에서 초기화 및 푸시
PC 터미널에서:

bash
코드 복사
cd ~/Downloads/ros2_backup  # 복사해온 폴더로 이동

git init
git add .
git commit -m "Initial commit: V2X ROS2 system"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/v2x-ros2-system.git
git push -u origin main
⚠️ GitHub 비밀번호 대신 Personal Access Token을 사용해야 합니다.
(생성: https://github.com/settings/tokens)

🧩 5. 업로드 후 확인
GitHub 리포지토리 페이지에서

car_msgs/, car_comms/, car_planning/ 폴더가 보이면 OK.

systemd/README.md 도 확인.

README.md의 프로젝트 설명이 첫 화면에 표시됩니다.

🔄 6. 프로젝트 복원(다른 기기에서)
bash
코드 복사
git clone https://github.com/<YOUR_USERNAME>/v2x-ros2-system.git
cd v2x-ros2-system
docker build -t v2x-ros2-system .
또는 기존처럼:

bash
코드 복사
docker run -it --rm --net=host --ipc=host \
  -e ROS_DOMAIN_ID=10 \
  -v $(pwd):/ws \
  -v /opt/v2x:/opt/v2x:ro \
  -w /ws \
  ros:foxy-ros-base bash
⚙️ 7. systemd 서비스 복원
GitHub에서 가져온 후:

bash
코드 복사
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ros2_v2x.service v2x-alert-client.service
sudo systemctl start  ros2_v2x.service v2x-alert-client.service
🧭 참고
ros2_v2x.service → Docker 기반 ROS2 자동 실행

v2x-alert-client.service → UDP 클라이언트 자동 실행

v2x-gateway.service → 게이트웨이 브릿지 (선택)

systemd/README.md 에 상세 설명 포함

✅ 완료
이제 GitHub에 전체 프로젝트가 업로드되면

코드 백업

팀 협업

다른 장치로 재현
모두 한 번에 해결됩니다 🎉
