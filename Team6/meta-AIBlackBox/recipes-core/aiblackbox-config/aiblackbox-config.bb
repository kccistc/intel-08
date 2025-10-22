SUMMARY = "AI BlackBox default config"
LICENSE = "CLOSED"

SRC_URI = "file://config.json"
S = "${WORKDIR}"

inherit allarch

do_install() {
    install -d ${D}${sysconfdir}/aiblackbox
    install -m 0644 ${WORKDIR}/config.json ${D}${sysconfdir}/aiblackbox/config.json
}

FILES:${PN} += "${sysconfdir}/aiblackbox/config.json"

# ---- static wired IPv4 (installed by aiblackbox-config) ----
SRC_URI += "file://10-eth0-static.network"
RRECOMMENDS:${PN} += "systemd-networkd"

do_install:append() {
    # 네트워크 파일 설치
    install -Dm0644 ${WORKDIR}/20-static-wired.network \
        ${D}${sysconfdir}/systemd/network/20-static-wired.network

    # systemd-networkd 부팅 자동 활성화(오프라인 enable)
    install -d ${D}${sysconfdir}/systemd/system/multi-user.target.wants
    ln -sf ${systemd_unitdir}/system/systemd-networkd.service \
        ${D}${sysconfdir}/systemd/system/multi-user.target.wants/systemd-networkd.service

    # 다른 네트워크 매니저들과 충돌 방지(있으면 마스킹)
    install -d ${D}${sysconfdir}/systemd/system
    for s in dhcpcd.service connman.service NetworkManager.service; do
        ln -sf /dev/null ${D}${sysconfdir}/systemd/system/$s
    done
}

# 패키징에 포함
FILES:${PN} += " \
  ${sysconfdir}/systemd/network/20-static-wired.network \
  ${sysconfdir}/systemd/system/multi-user.target.wants/systemd-networkd.service \
  ${sysconfdir}/systemd/system/dhcpcd.service \
  ${sysconfdir}/systemd/system/connman.service \
  ${sysconfdir}/systemd/system/NetworkManager.service \
"
# ---- end static wired IPv4 ----
