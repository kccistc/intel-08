
do_install:append() {
    # --- firmware(cmdline.txt) 계열 ---
    for C in \
        ${D}${sysconfdir}/firmware/cmdline.txt \
        ${D}/boot/cmdline.txt \
        ${D}/boot/firmware/cmdline.txt ; do
        [ -f "$C" ] || continue
        # root=를 /dev/mmcblk0p2로 통일
        sed -i -e 's#root=[^ ]*#root=/dev/mmcblk0p2#' "$C" || true
        # console=tty1 없으면 추가
        grep -q 'console=tty' "$C" || sed -i -e 's#$# console=tty1#' "$C"
        # KMS 해상도(800x480) 없으면 추가
        grep -q 'video=HDMI-A-1' "$C" || sed -i -e 's#$# video=HDMI-A-1:800x480@60D#' "$C"
    done

    # --- U-Boot/extlinux 계열 ---
    if [ -f ${D}/boot/extlinux/extlinux.conf ]; then
        sed -i -e 's#\(APPEND .*\)root=[^ ]*#\1root=/dev/mmcblk0p2#' ${D}/boot/extlinux/extlinux.conf || true
        grep -q 'console=tty' ${D}/boot/extlinux/extlinux.conf || \
          sed -i -e 's#^\([[:space:]]*APPEND .*\)#\1 console=tty1#' ${D}/boot/extlinux/extlinux.conf
        grep -q 'video=HDMI-A-1' ${D}/boot/extlinux/extlinux.conf || \
          sed -i -e 's#^\([[:space:]]*APPEND .*\)#\1 video=HDMI-A-1:800x480@60D#' ${D}/boot/extlinux/extlinux.conf
    fi
}

# 우리가 만진 파일들이 패키지에 반드시 포함되도록
FILES:${PN}:append = " /boot/cmdline.txt /boot/firmware/cmdline.txt /boot/extlinux/extlinux.conf ${sysconfdir}/firmware/cmdline.txt "
