require recipes-core/images/core-image-base.bb

EXTRA_IMAGE_FEATURES += "ssh-server-openssh debug-tweaks"
LICENSE_FLAGS_ACCEPTED += "commercial"

IMAGE_INSTALL:append = " \
  v4l-utils \
  gstreamer1.0 \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-base-videoconvert \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-good-jpeg \
  gstreamer1.0-plugins-good-isomp4 \
  gstreamer1.0-plugins-good-video4linux2 \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-bad-videoparsersbad \
  gstreamer1.0-plugins-bad-v4l2codecs \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-plugins-ugly-x264 \
  gstreamer1.0-libav \
  gstreamer1.0-plugins-bad-kms \
  x264 \
  opencv opencv-apps \
  cjson nlohmann-json \
  python3 python3-core python3-venv python3-pip python3-setuptools python3-wheel \
  python3-numpy python3-pyyaml python3-can \
  can-utils iproute2 i2c-tools \
  tcpdump ca-certificates \
  libdrm libdrm-tests \
  libgbm libegl-mesa libgles2-mesa \
  xserver-xorg xinit xauth \
  mesa \
  qtbase qtbase-plugins \
  libx11 libx11-xcb libxcb libxext libxrender libxi libxrandr \
  wayland libxkbcommon \
  gtk+3 cairo \
  libxcrypt libffi sqlite3 bzip2 xz readline gdbm keyutils krb5 \
  libss libext2fs libe2p \
  linux-firmware-rpidistro-bcm43455 \
  hailo-pci hailo-firmware hailort \
  librealsense2 librealsense2-debug-tools \
  aiblackbox-can aiblackbox-diagnostics libhardware \
  gobject-introspection gobject-introspection-dev \
  glib-2.0 glib-2.0-dev \
  libffi-dev \
  cairo-dev \
  pkgconfig gcc make meson ninja \
  file glibc-utils \
"
# (선택) Hailo 드라이버 자동 로드 - 드라이버 모듈명이 hailo_pci 인 경우
KERNEL_MODULE_AUTOLOAD += " hailo_pci "

# (SPI CAN 자동 로드/설정은 기존 그대로 유지)
IMAGE_CLASSES        += "rpi-config"
IMAGE_FSTYPES        += "rpi-sdimg"
KERNEL_MODULE_AUTOLOAD += " mcp251x can can-raw can-dev "

# (선택) SPI0 MCP2515 오버레이 패키지가 따로면 유지
IMAGE_INSTALL:append = " aibb-spi0-mcp2515  python38-bin python38-bin fontconfig ttf-dejavu-sans ttf-dejavu-sans-mono ttf-dejavu-serif xserver-xorg libxkbcommon xkeyboard-config xcb-util xcb-util-image xcb-util-keysyms xcb-util-wm fontconfig ttf-dejavu-sans ttf-dejavu-sans-mono ttf-dejavu-serif libdrm libgbm libegl-mesa libgles2-mesa libinput"
EXTRA_USERS_PARAMS:append = " usermod -p '$(openssl passwd -6 raspberry)' root; "
CMDLINE:append = " console=tty1 video=HDMI-A-1:800x480@60D"
IMAGE_INSTALL:append = " python38-bin python38-bin fontconfig ttf-dejavu-sans ttf-dejavu-sans-mono ttf-dejavu-serif xserver-xorg libxkbcommon xkeyboard-config xcb-util xcb-util-image xcb-util-keysyms xcb-util-wm fontconfig ttf-dejavu-sans ttf-dejavu-sans-mono ttf-dejavu-serif libdrm libgbm libegl-mesa libgles2-mesa libinput"
