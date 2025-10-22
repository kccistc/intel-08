#include <string.h>
#include "hardware.h"

// RGB24 기준 직사각형 경계선 그리기
void graphics_draw_rectangle(FrameBuffer* frame, int x, int y, int w, int h, int thickness, unsigned int color) {
    if (!frame || !frame->data) return;
    if (thickness < 1) thickness = 1;
    int W = frame->width, H = frame->height;
    unsigned char r = (color >> 16) & 0xFF;
    unsigned char g = (color >> 8)  & 0xFF;
    unsigned char b = (color)       & 0xFF;

    for (int t = 0; t < thickness; ++t) {
        int yy1 = y + t, yy2 = y + h - 1 - t;
        if (yy1 >= 0 && yy1 < H) {
            for (int xx = x; xx < x + w; ++xx) {
                if (xx < 0 || xx >= W) continue;
                int idx = (yy1 * W + xx) * 3;
                frame->data[idx+0] = r; frame->data[idx+1] = g; frame->data[idx+2] = b;
            }
        }
        if (yy2 >= 0 && yy2 < H) {
            for (int xx = x; xx < x + w; ++xx) {
                if (xx < 0 || xx >= W) continue;
                int idx = (yy2 * W + xx) * 3;
                frame->data[idx+0] = r; frame->data[idx+1] = g; frame->data[idx+2] = b;
            }
        }
        int xx1 = x + t, xx2 = x + w - 1 - t;
        if (xx1 >= 0 && xx1 < W) {
            for (int yy = y; yy < y + h; ++yy) {
                if (yy < 0 || yy >= H) continue;
                int idx = (yy * W + xx1) * 3;
                frame->data[idx+0] = r; frame->data[idx+1] = g; frame->data[idx+2] = b;
            }
        }
        if (xx2 >= 0 && xx2 < W) {
            for (int yy = y; yy < y + h; ++yy) {
                if (yy < 0 || yy >= H) continue;
                int idx = (yy * W + xx2) * 3;
                frame->data[idx+0] = r; frame->data[idx+1] = g; frame->data[idx+2] = b;
            }
        }
    }
}

void graphics_draw_text(FrameBuffer* frame, const char* text, int x, int y, int font_size, unsigned int color) {
    (void)frame; (void)text; (void)x; (void)y; (void)font_size; (void)color; // 스텁
}

int lcd_display_frame(const FrameBuffer* frame) {
    (void)frame; // DRM/KMS 구현은 후속
    return 0;
}

