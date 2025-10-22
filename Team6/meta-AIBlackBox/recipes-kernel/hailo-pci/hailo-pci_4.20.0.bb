SUMMARY = "Hailo-8 PCIe kernel driver (v4.20.0)"
HOMEPAGE = "https://github.com/hailo-ai/hailort-drivers"
LICENSE = "GPL-2.0-only"
LIC_FILES_CHKSUM = "file://LICENSE;md5=39bba7d2cf0ba1036f2a6e2be52fe3f0"

PV = "4.20.0"

# Hailo-8은 master가 아님. 태그 커밋을 nobranch=1로 고정.
SRC_URI = "git://github.com/hailo-ai/hailort-drivers.git;protocol=https;nobranch=1"
SRCREV  = "d1af769eb1d8074c5a0151a37b22b46bd483e5a7"

S = "${WORKDIR}/git"

inherit module

do_compile() {
    oe_runmake -C ${STAGING_KERNEL_DIR} M=${S}/linux/pcie modules
}

do_install() {
    oe_runmake -C ${STAGING_KERNEL_DIR} M=${S}/linux/pcie \
        INSTALL_MOD_PATH=${D} INSTALL_MOD_DIR=extra modules_install
}
