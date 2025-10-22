FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SUMMARY = "HailoRT runtime (v4.20.0)"
HOMEPAGE = "https://github.com/hailo-ai/hailort"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${WORKDIR}/LICENSE.MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

PV = "4.20.0"
SRC_URI = "git://github.com/hailo-ai/hailort.git;protocol=https;nobranch=1 \
           file://LICENSE.MIT"
SRCREV = "542ba8f3cd95ed85175083ee4add00167c50f668"

S = "${WORKDIR}/git"
B = "${WORKDIR}/build"

inherit cmake pkgconfig

# protoc(native) + protobuf(target) + ninja + git
DEPENDS = "protobuf-native protobuf zlib ninja-native git-native"

# FetchContent로 외부 서브프로젝트 받게 허용
do_configure[network] = "1"
do_compile[network]   = "1"
OECMAKE_GENERATOR = "Unix Makefiles"

EXTRA_OECMAKE = "\
  -DCMAKE_BUILD_TYPE=Release \
  -DHAILO_BUILD_EXAMPLES=OFF \
  -DHAILO_BUILD_EMULATOR=OFF \
  -DHAILO_BUILD_GSTREAMER=OFF \
  -DHAILO_BUILD_TESTS=OFF \
  -DHAILO_BUILD_PYHAILORT=OFF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
  -DProtobuf_PROTOC_EXECUTABLE=${STAGING_BINDIR_NATIVE}/protoc \
"

# ---------- Packaging ----------
PACKAGES = "${PN} ${PN}-libs ${PN}-cli ${PN}-dev ${PN}-dbg"

# 메타 패키지(hailort) → 빈 패키지, 설치 시 libs/cli를 끌어오게
ALLOW_EMPTY:${PN} = "1"
FILES:${PN} = ""
RDEPENDS:${PN} = "${PN}-libs ${PN}-cli"

# ★ dev-so QA 해결: libs에는 버전 so만, dev에는 비버전 so(심볼릭)
FILES:${PN}-libs = "${libdir}/libhailort.so.*"
FILES:${PN}-dev  = "${includedir}/hailo/* ${libdir}/libhailort.so ${libdir}/pkgconfig/* ${libdir}/cmake/HailoRT*"

# CLI
FILES:${PN}-cli  = "${bindir}/hailortcli"
RDEPENDS:${PN}-cli = "${PN}-libs"

# 혹시 외부가 런타임에 'hailort'를 요구해도 libs가 제공
RPROVIDES:${PN}-libs += "hailort"
