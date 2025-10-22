#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <linux/can.h>
#include <linux/can/raw.h>

#include "hardware.h"

static int s_can_fd = -1;

int can_init(int bitrate) {
    (void)bitrate; // bitrate는 systemd 유닛에서 처리
    s_can_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s_can_fd < 0) { perror("socket(PF_CAN)"); return -1; }

    struct ifreq ifr = {0};
    strncpy(ifr.ifr_name, "can0", IFNAMSIZ-1);
    if (ioctl(s_can_fd, SIOCGIFINDEX, &ifr) < 0) {
        perror("SIOCGIFINDEX"); close(s_can_fd); s_can_fd = -1; return -1;
    }
    struct sockaddr_can addr = {0};
    addr.can_family = AF_CAN; addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(s_can_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind(can0)"); close(s_can_fd); s_can_fd = -1; return -1;
    }
    int flags = fcntl(s_can_fd, F_GETFL, 0);
    fcntl(s_can_fd, F_SETFL, flags | O_NONBLOCK);
    return 0;
}

int can_send_message(const CANMessage* msg) {
    if (s_can_fd < 0 || !msg) return -1;
    struct can_frame frame = {0};
    frame.can_id = msg->id; frame.can_dlc = msg->dlc;
    memcpy(frame.data, msg->data, frame.can_dlc);
    int n = write(s_can_fd, &frame, sizeof(frame));
    return (n == (int)sizeof(frame)) ? 0 : -1;
}

int can_receive_message(CANMessage* msg) {
    if (s_can_fd < 0 || !msg) return -1;
    struct can_frame frame;
    int n = read(s_can_fd, &frame, sizeof(frame));
    if (n < 0) return 0;          // 논블로킹: 수신 없음
    if (n < (int)sizeof(frame)) return -1;
    msg->id = frame.can_id; msg->dlc = frame.can_dlc;
    memcpy(msg->data, frame.data, frame.can_dlc);
    return 1;
}

void can_close() {
    if (s_can_fd >= 0) { close(s_can_fd); s_can_fd = -1; }
}

