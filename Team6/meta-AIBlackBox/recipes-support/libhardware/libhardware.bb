SUMMARY = "AI Blackbox hardware lib"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://COPYING.MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit cmake pkgconfig

SRC_URI = "file://CMakeLists.txt \
           file://include/hardware.h \
           file://src/hardware.c \
           file://src/can_socketcan.c \
           file://src/graphics_stub.c \
           file://src/camera_realsense.cpp \
           file://src/storage.c \
           file://COPYING.MIT \
          "

S = "${WORKDIR}"

DEPENDS += "gstreamer1.0 gstreamer1.0-plugins-base librealsense2 cjson"
DEPENDS += "libpcap"

CFLAGS:append   = " -fdebug-prefix-map=${WORKDIR}=/usr/src/debug/${PN}/${PV}-${PR}"
CXXFLAGS:append = " -fdebug-prefix-map=${WORKDIR}=/usr/src/debug/${PN}/${PV}-${PR}"

FILES:${PN}     += "${libdir}/libhardware.so*"
FILES:${PN}-dev += "${includedir}/hardware.h"
