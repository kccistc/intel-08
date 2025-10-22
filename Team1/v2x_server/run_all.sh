#!/bin/bash

# ==================================================================
# V2X, CARLA, Streamlit 서버 동시 실행 스크립트
# ==================================================================

# --- 1. 설정: 경로 정의 ---

# 공통 작업 디렉토리
WORK_DIR="/home/ksj/workspace/git-practice/intel-08/Team1/v2x_ros2_system/v2x_server"

# 가상 환경(venv) 경로
VENV_WORKSPACE="/home/ksj/workspace/.venv"
VENV_CARLA="/home/ksj/CARLA_0.9.16/.venv"

# --- 2. 실행 ---

echo "모든 V2X 시스템 프로세스를 백그라운드로 실행합니다..."
echo "작업 디렉토리: $WORK_DIR"
echo "---"

# 작업 디렉토리로 이동
cd "$WORK_DIR"

# 프로세스 1: watch_and_broadcast.py
echo "1. watch_and_broadcast.py 시작 중... (Venv: $VENV_WORKSPACE)"
"$VENV_WORKSPACE/bin/python3" watch_and_broadcast.py &
PID1=$!

# 프로세스 2: carla_with_flask.py
echo "2. carla_with_flask.py 시작 중... (Venv: $VENV_CARLA)"
"$VENV_CARLA/bin/python3" carla_with_flask.py &
PID2=$!

# 프로세스 3: streamlit_app.py
echo "3. streamlit_app.py 시작 중... (Venv: $VENV_WORKSPACE)"
"$VENV_WORKSPACE/bin/streamlit" run streamlit_app.py &
PID3=$!

# --- 3. 완료 ---
echo "---"
echo "모든 프로세스가 시작되었습니다."
echo "  - V2X Broadcaster (PID: $PID1)"
echo "  - CARLA Flask (PID: $PID2)"
echo "  - Streamlit App (PID: $PID3)"
echo ""
echo "프로세스들을 한 번에 종료하려면 아래 명령어를 사용하세요:"
echo "kill $PID1 $PID2 $PID3"