SUMMARY = "Device Tree Overlay: MCP2515 on SPI0 (CS1=GPIO8)"
LICENSE = "CLOSED"
PACKAGE_ARCH = "${MACHINE_ARCH}"
COMPATIBLE_MACHINE = "(^raspberrypi.*|^rpi.*)"

DEPENDS += "dtc-native"

SRC_URI = "file://aibb-spi0-mcp2515.dts"

S = "${WORKDIR}"
B = "${WORKDIR}/build"

do_compile() {
    install -d ${B}
    ${STAGING_BINDIR_NATIVE}/dtc -I dts -O dtb -@ \
        -o ${B}/aibb-spi0-mcp2515.dtbo \
        ${WORKDIR}/aibb-spi0-mcp2515.dts
}

do_install() {
    install -d ${D}/boot/overlays
    install -m 0644 ${B}/aibb-spi0-mcp2515.dtbo ${D}/boot/overlays/
}

FILES:${PN} += "/boot/overlays/aibb-spi0-mcp2515.dtbo"
