cmake_minimum_required(VERSION 3.10)
project(hardware C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(PkgConfig REQUIRED)

# pkg-config 모듈 찾기 (cjson은 배포에 따라 이름 다름)
pkg_check_modules(GST     REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP  REQUIRED gstreamer-app-1.0)
pkg_check_modules(DRM     REQUIRED libdrm)
pkg_check_modules(RS2     REQUIRED realsense2)
pkg_check_modules(CJSON   QUIET    cjson)
pkg_check_modules(PCAP REQUIRED libpcap)

if(NOT CJSON_FOUND)
  pkg_check_modules(CJSON REQUIRED libcjson)
endif()

add_library(hardware SHARED
  src/can_socketcan.c
  src/camera_realsense.cpp
  src/graphics_stub.c
  src/storage.c
)

set_target_properties(hardware PROPERTIES VERSION 1.0.0 SOVERSION 1)

target_include_directories(hardware
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${GST_INCLUDE_DIRS} ${GSTAPP_INCLUDE_DIRS}
          ${DRM_INCLUDE_DIRS} ${CJSON_INCLUDE_DIRS}
          ${RS2_INCLUDE_DIRS} ${PCAP_INCLUDE_DIRS}
)

# pkg-config가 주는 기타 컴파일 플래그(예: -pthread, GLib 경로 등) 반영
target_compile_options(hardware PRIVATE
  ${GST_CFLAGS_OTHER} ${GSTAPP_CFLAGS_OTHER} ${PCAP_CFLAGS_OTHER}
  ${DRM_CFLAGS_OTHER} ${CJSON_CFLAGS_OTHER} ${RS2_CFLAGS_OTHER}
)

target_link_libraries(hardware
  ${GST_LIBRARIES} ${GSTAPP_LIBRARIES} ${PCAP_LIBRARIES}
  ${DRM_LIBRARIES} ${CJSON_LIBRARIES} ${RS2_LIBRARIES}
)

include(GNUInstallDirs)
install(TARGETS hardware
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/hardware.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
