#include <stdlib.h>
#include <string.h>
#include "hardware.h"

// 현재는 빈 프레임 리턴(검정). 추후 librealsense2로 교체.
FrameBuffer* camera_get_frame() {
    const int w = 640, h = 480, bpp = 3;
    int size = w * h * bpp;
    unsigned char* buf = (unsigned char*)malloc(size);
    if (!buf) return NULL;
    memset(buf, 0x00, size);

    FrameBuffer* fb = (FrameBuffer*)malloc(sizeof(FrameBuffer));
    if (!fb) { free(buf); return NULL; }
    fb->data = buf; fb->width = w; fb->height = h; fb->size = size; fb->private_data = NULL;
    return fb;
}

void camera_release_frame(FrameBuffer* frame) {
    if (!frame) return;
    if (frame->data) free(frame->data);
    free(frame);
}

