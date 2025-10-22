#!/bin/sh
set -x
export PATH=/usr/sbin:/sbin:/usr/bin:/bin:$PATH

echo "--- KERNEL/BOOT ---"
uname -a
printf "cmdline: %s\n" "$(cat /proc/cmdline 2>/dev/null || echo N/A)"
if [ -f /boot/config.txt ]; then
  echo "--- /boot/config.txt (key lines) ---"
  grep -E '^(dtparam|dtoverlay|camera_|hdmi_|hdmi_cvt|enable_uart)' /boot/config.txt || true
fi

echo "--- DRM/KMS ---"
ls -l /dev/dri || true
modetest -M vc4 -c || true

echo "--- USB/RealSense ---"
lsusb || true
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices || true
fi

echo "--- CAN ---"
ip -details link show can0 || true
if command -v candump >/dev/null 2>&1; then
  candump -L -n 10 -t a can0 || true
fi
journalctl -u can0.service --no-pager -n 50 || true

echo "--- DT / overlays ---"
grep -Rsn "mcp2515" /proc/device-tree 2>/dev/null || true
ls -l /boot/overlays | head || true

echo "--- DMESG (vc4 / spi / can / mcp2515) ---"
dmesg | grep -Ei "vc4|v3d|spi|can|mcp2515" || true
