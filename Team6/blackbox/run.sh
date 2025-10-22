# #!/bin/bash



# SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# echo "--- Blackbox Application Starting ---"

# # 1. 공유 라이브러리 경로 설정
# export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib"
# echo "Library path set to: ${LD_LIBRARY_PATH}"

# # 2. 메인 애플리케이션 실행
# "${SCRIPT_DIR}/bin/blackbox_main"

# echo "--- Blackbox Application Finished ---"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "--- Blackbox Application Starting ---"

# 0) SocketCAN(can0) 재설정: down -> bitrate 500000 -> up
configure_can() {
  echo "[CAN] Reconfiguring can0 to 500000 bps..."
  # root가 아니면 sudo로 실행
  SUDO=""
  if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
  fi

  # ip 존재 확인
  if ! command -v ip >/dev/null 2>&1; then
    echo "[CAN][ERROR] 'ip' command not found. Install iproute2."
    exit 1
  fi

  # down은 실패해도 계속 진행
  $SUDO ip link set can0 down || true
  # classic CAN 500kbps 설정
  $SUDO ip link set can0 type can bitrate 500000
  # 다시 up
  $SUDO ip link set can0 up

  # 상태 표시(옵션)
  $SUDO ip -details link show can0 | sed 's/^/  /'
  echo "[CAN] can0 is up @ 500000 bps"
}

configure_can

# 1) 공유 라이브러리 경로 설정
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib"
echo "Library path set to: ${LD_LIBRARY_PATH}"

# 2) 메인 애플리케이션 실행
"${SCRIPT_DIR}/bin/blackbox_main"

echo "--- Blackbox Application Finished ---"