# =================================================================
#    libhardware Makefile (cJSON Submodule + GStreamer via pkg-config)
# =================================================================

CC         ?= gcc
CXX        ?= g++
PKG_CONFIG ?= pkg-config

CFLAGS   ?= -Wall -O2 -fPIC
CXXFLAGS ?= $(CFLAGS)
# 환경에서 온 LDFLAGS가 있어도 공유라이브러리 플래그가 항상 붙도록 분리
SO_LDFLAGS = -shared
LDFLAGS  ?=
LDLIBS   ?=

INC_DIRS  = -I./include -I../vendor/cJSON
CFLAGS   += $(INC_DIRS)
CXXFLAGS += $(INC_DIRS)

GST_PKGS   = gstreamer-1.0 gstreamer-app-1.0 glib-2.0
GST_CFLAGS := $(shell $(PKG_CONFIG) --cflags $(GST_PKGS) 2>/dev/null)
GST_LIBS   := $(shell $(PKG_CONFIG) --libs   $(GST_PKGS) 2>/dev/null)

ifneq ($(strip $(GST_CFLAGS)$(GST_LIBS)),)
  CFLAGS   += $(GST_CFLAGS)
  CXXFLAGS += $(GST_CFLAGS)
  LDLIBS   += $(GST_LIBS)
else
  LDLIBS   += -lgstreamer-1.0 -lgstapp-1.0 -lglib-2.0 -lgobject-2.0 -lgthread-2.0 -pthread
endif

SRC_DIR    = src
BUILD_DIR  = ../build
OBJ_DIR_C  = $(BUILD_DIR)/obj/c
OBJ_DIR_CXX= $(BUILD_DIR)/obj/cpp
TARGET     = $(BUILD_DIR)/lib/libhardware.so

SRCS_C     = $(wildcard $(SRC_DIR)/*.c)
SRCS_CXX   = $(wildcard $(SRC_DIR)/*.cpp)
VENDOR_CJSON_SRC = ../vendor/cJSON/cJSON.c

OBJS_C     = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR_C)/src/%.o,$(SRCS_C))
OBJS_CXX   = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR_CXX)/src/%.o,$(SRCS_CXX))
OBJ_CJSON  = $(OBJ_DIR_C)/vendor/cJSON/cJSON.o
OBJECTS    = $(OBJS_C) $(OBJS_CXX) $(OBJ_CJSON)
LDLIBS    += -lm

ifeq ($(strip $(SRCS_CXX)),)
  LINKER = $(CC)
else
  LINKER = $(CXX)
endif

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Linking Library: $@"
	@mkdir -p $(@D)
	# 핵심: -shared를 무조건 앞에 붙인다
	$(LINKER) $(SO_LDFLAGS) $(LDFLAGS) -Wl,-soname,libhardware.so -o $@ $^ $(LDLIBS)

$(OBJ_DIR_C)/src/%.o: $(SRC_DIR)/%.c
	@echo "CC  $<"
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR_CXX)/src/%.o: $(SRC_DIR)/%.cpp
	@echo "CXX $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_CJSON): $(VENDOR_CJSON_SRC)
	@echo "CC  $<"
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<
	
clean:
	@echo "Cleaning up library build files..."
	rm -rf $(BUILD_DIR)/obj $(BUILD_DIR)/lib
