# =================================================================
#           Blackbox Project - 최상위 Makefile (최종 버전)
# =================================================================

# --- 배포 기본값 설정 ---
PI_USER ?= pi
PI_IP ?= 10.10.14.61

# --- 크로스 컴파일러 설정 ---
CROSS_COMPILE = aarch64-linux-gnu-

# 컴파일러 변수 정의 (네이티브 빌드 시 기본값)
CC ?= gcc

# .PHONY: 가상 목표 선언
.PHONY: all cross lib app run deploy clean

# 'make' 또는 'make all': 우분투 PC에서 테스트하기 위한 네이티브 빌드
all: lib app

# 'make cross': 라즈베리파이용으로 크로스 컴파일하는 전용 목표
cross:
	@echo "--- Cross-compiling for Raspberry Pi (ARM64) ---"
	$(MAKE) all CC=$(CROSS_COMPILE)gcc

# 라이브러리 빌드
lib:
	$(MAKE) -C libhardware CC=$(CC)

# 애플리케이션 빌드
app:
	$(MAKE) -C app CC=$(CC)

# 라즈베리파이로 배포하는 규칙
deploy: cross
	@echo "--- Deploying to Raspberry Pi ---"
	./deploy.sh $(PI_USER) $(PI_IP)

# 실행 규칙 (라즈베리파이에서만 사용)
run:
	@echo "This target should be run on the Raspberry Pi."

# 정리 규칙
clean:
	@echo "--- Cleaning up the project ---"
	$(MAKE) -C libhardware clean
	$(MAKE) -C app clean
	rm -rf build