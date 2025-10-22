RPI_KERNEL_CMDLINE = "dwc_otg.lpm_enable=0 root=/dev/mmcblk0p2 rootfstype=ext4 rootwait"
RPI_EXTRA_CONFIG:append = " \
disable_overscan=1\n \
hdmi_force_hotplug=1\n \
hdmi_group=2\n \
hdmi_mode=87\n \
hdmi_cvt=800 480 60 6 0 0 0\n \
hdmi_drive=2\n \
dtoverlay=vc4-kms-v3d-pi5\n \
dtoverlay=vc4-kms-v3d\n \
dtparam=spi=on\n \
dtoverlay=mcp2515-can0,oscillator=8000000,interrupt=25,spimaxfrequency=10000000\n \
dtparam=pciex1\n \
"
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"