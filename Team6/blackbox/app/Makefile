# =================================================================
#               App (blackbox_main) 빌드용 Makefile (최종본)
# =================================================================

# --- 컴파일러 및 옵션 정의 ---
# 최상위 Makefile에서 CC 변수를 받아 사용하며, 없으면 gcc를 기본값으로 사용합니다.
CC ?= gcc

# CFLAGS: 컴파일 옵션
# -Wall: 모든 경고 메시지를 활성화합니다.
# -I../libhardware/include: 'hardware.h' 헤더 파일을 찾기 위해 경로를 추가합니다.
# -I../vendor/cJSON: 'cJSON.h' 헤더 파일을 찾기 위해 경로를 추가합니다.
CFLAGS = -Wall -I../libhardware/include -I../vendor/cJSON

# LDFLAGS: 링커 옵션
# -L../build/lib: 'libhardware.so' 라이브러리 파일이 있는 경로를 지정합니다.
# -lhardware: 'libhardware.so' 라이브러리와 링크하도록 지정합니다.
LDFLAGS = -L../build/lib 

LDLIBS  = -lhardware -lm  

# --- 소스 및 결과물 경로 정의 ---
SRC = src/main.c
BUILD_DIR = ../build
TARGET = $(BUILD_DIR)/bin/blackbox_main

# 의존성: App을 빌드하기 전에 반드시 이 라이브러리 파일이 존재해야 합니다.
LIB_DEPENDENCY = $(BUILD_DIR)/lib/libhardware.so


# --- 빌드 규칙 정의 ---

# 'make' 또는 'make all' 실행 시 기본적으로 수행할 목표
all: $(TARGET)

# 최종 실행 파일($(TARGET))을 생성하는 규칙
# $(SRC) 파일과 $(LIB_DEPENDENCY) 파일에 의존합니다.
# 즉, main.c가 수정되거나 libhardware.so가 새로 만들어지면 다시 빌드됩니다.
$(TARGET): $(SRC) $(LIB_DEPENDENCY)
	@echo "Compiling and linking application: $@"
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS) $(LDLIBS)

# 'make clean' 실행 시 컴파일 결과물을 삭제하는 규칙
clean:
	@echo "Cleaning up application build files..."
	rm -f $(TARGET)

# 가상 목표(PHONY) 선언
.PHONY: all clean