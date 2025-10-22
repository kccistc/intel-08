SUMMARY = "BSP diagnostics script"
LICENSE = "CLOSED"
SRC_URI = "file://bsp_diag.sh"
S = "${WORKDIR}"
RDEPENDS:${PN} = "bash coreutils grep iproute2 can-utils libdrm-tests usbutils"
do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/bsp_diag.sh ${D}${bindir}/bsp_diag.sh
}

