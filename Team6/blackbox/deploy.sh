#!/bin/bash

# =================================================================
#    Blackbox Project - Raspberry Pi 배포 스크립트 (매개변수 버전)
# =================================================================

# --- 1. 입력 매개변수 확인 ---
# 스크립트 실행 시 필요한 인자(username, ip_address)가 2개가 아니면 사용법을 안내하고 종료합니다.
if [ "$#" -ne 2 ]; then
    echo "사용법: $0 <사용자이름> <IP주소>"
    echo "예시: $0 pi 10.10.14.61"
    exit 1
fi

# --- 2. 설정 ---
# 입력받은 매개변수로 변수를 설정합니다.
USERNAME=$1
IP_ADDRESS=$2
PI_USER_HOST="${USERNAME}@${IP_ADDRESS}"
DEST_DIR="~/blackbox"

echo ">>> 라즈베리파이 배포를 시작합니다. 대상: ${PI_USER_HOST}"

# --- 3. 원격 폴더 생성 ---
echo ">>> 원격 디렉터리 구조를 생성합니다..."
ssh ${PI_USER_HOST} "mkdir -p ${DEST_DIR}/bin ${DEST_DIR}/lib ${DEST_DIR}/ai ${DEST_DIR}/assets"
echo "    ...완료."

# --- 4. 파일 전송 ---
echo ">>> 컴파일된 파일들을 전송합니다..."

echo "    - 실행 파일 (blackbox_main) 전송 중..."
scp build/bin/blackbox_main ${PI_USER_HOST}:${DEST_DIR}/bin/

echo "    - 공유 라이브러리 (libhardware.so) 전송 중..."
scp build/lib/libhardware.so ${PI_USER_HOST}:${DEST_DIR}/lib/

echo "    - AI 서버 파일 전송 중..."
scp -r ai/* ${PI_USER_HOST}:${DEST_DIR}/ai/

echo "    - Assets 파일 전송 중..."
scp -r assets/* ${PI_USER_HOST}:${DEST_DIR}/assets/

echo "    - 실행 스크립트 (run.sh) 전송 중..."
scp run.sh ${PI_USER_HOST}:${DEST_DIR}/

echo "    - 원격 스크립트 실행 권한 부여 중..."
ssh ${PI_USER_HOST} "chmod +x ${DEST_DIR}/run.sh"

echo "    ...완료."
echo ""
echo ">>> 모든 파일 전송 완료!"