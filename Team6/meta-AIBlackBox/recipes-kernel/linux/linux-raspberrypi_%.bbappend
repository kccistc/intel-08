
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

# 커널 설정 프래그먼트 추가
SRC_URI:append = " file://can-mcp251x.cfg"
KERNEL_CONFIG_FRAGMENTS:append = " can-mcp251x.cfg"
