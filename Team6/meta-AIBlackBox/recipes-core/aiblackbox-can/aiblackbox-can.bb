SUMMARY = "CAN bring-up service (MCP2515 -> can0)"
LICENSE = "CLOSED"
SRC_URI = "file://can0.service"
S = "${WORKDIR}"
RDEPENDS:${PN} = "iproute2 can-utils"
inherit systemd
do_install() {
    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/can0.service ${D}${systemd_system_unitdir}/
}
SYSTEMD_SERVICE:${PN} = "can0.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

