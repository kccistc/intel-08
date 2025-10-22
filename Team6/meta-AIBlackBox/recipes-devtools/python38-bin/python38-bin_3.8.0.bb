
SUMMARY = "CPython 3.8.0 prebuilt bundle (aarch64) with compat libs"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = "file://python38-aarch64.tar.xz;unpack=0"
S = "${WORKDIR}"

# 패키징/QA 튜닝
INHIBIT_PACKAGE_STRIP = "1"
INHIBIT_PACKAGE_DEBUG_SPLIT = "1"
# cgi.py 등의 문서/예제 문자열 때문에 생기는 절대경로 의존 경고 무시
INSANE_SKIP:${PN} += "file-rdeps already-stripped ldflags"

do_install() {
    # tar payload: /opt/python-3.8, /opt/_internal/*
    tar -C ${D} -xJf ${WORKDIR}/python38-aarch64.tar.xz

    # 번들 so들을 표준 libdir로 이동 → RPM이 soname Provides 인식
    install -d ${D}${libdir}/python38-compat
    if [ -d ${D}/opt/_internal/lib64 ]; then
        cp -a ${D}/opt/_internal/lib64/*.so* ${D}${libdir}/python38-compat/ || true
        rm -rf ${D}/opt/_internal
    fi

    # 파이썬 래퍼로 compat 경로 주입
    mv ${D}/opt/python-3.8/bin/python3.8 ${D}/opt/python-3.8/bin/python3.8.bin
    cat > ${D}/opt/python-3.8/bin/python3.8 <<'WRAP'
#!/bin/sh
export LD_LIBRARY_PATH=/usr/lib/python38-compat:${LD_LIBRARY_PATH:-}
exec /opt/python-3.8/bin/python3.8.bin "$@"
WRAP
    chmod 0755 ${D}/opt/python-3.8/bin/python3.8

    # 용량/의존성 줄이기(선택) - 테스트 제거
    rm -rf ${D}/opt/python-3.8/lib/python3.8/test || true
}

FILES:${PN} += " \
    /opt/python-3.8 \
    ${libdir}/python38-compat \
"

do_install:append() {
    rm -f ${D}/opt/python-3.8/lib/*.a || true
    rm -f ${D}/opt/python-3.8/lib/python3.8/config-*/libpython*.a || true
}

# ======= 중요: RPM 자동 의존성/파일의존성/쉐어드라이브 스캔 끄기 =======
# shlibs/soname 스캔과 filedeps 스캔을 끄면 Requires가 생기지 않아 dnf 충돌이 사라짐
INHIBIT_PACKAGE_SHLIBS = "1"
SKIP_FILEDEPS = "1"

# 혹시 남는 QA 잡음을 막기 위한 최소 설정
INSANE_SKIP:${PN} += " file-rdeps staticdev "

# (명시적으로 런타임 의존성 비우기)
RDEPENDS:${PN} = ""
RRECOMMENDS:${PN} = ""
