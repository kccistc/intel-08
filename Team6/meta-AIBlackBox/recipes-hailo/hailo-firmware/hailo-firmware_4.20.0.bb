FILESEXTRAPATHS:prepend := "${THISDIR}/files:"
SUMMARY = "Hailo-8 firmware blob (v4.20.0)"
LICENSE = "CLOSED"
LIC_FILES_CHKSUM = ""
S = "${WORKDIR}"

SRC_URI = "file://hailo8_fw_4.20.0.bin"

do_install() {
    install -d ${D}/lib/firmware/hailo
    install -m 0644 ${WORKDIR}/hailo8_fw_4.20.0.bin ${D}/lib/firmware/hailo/
    ln -sf hailo8_fw_4.20.0.bin ${D}/lib/firmware/hailo/hailo8_fw.bin
}

FILES:${PN} += "/lib/firmware/hailo/*"
# (선택) 라이선스 단계 스킵하고 싶으면 아래 한 줄 추가
do_populate_lic[noexec] = "1"
