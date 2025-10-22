#ifndef HARDWARE_H
#define HARDWARE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// ================= 1. 공통 및 초기화 API =================

// ================= 2. 카메라 API =================
typedef struct {
    unsigned char* data; // RGB24
    int width;
    int height;
    size_t size;            // bytes
    void* private_data;  // 내부 상태 포인터(옵션)
} FrameBuffer;

FrameBuffer* camera_get_frame();
void camera_release_frame(FrameBuffer* frame);

// ================= 3. 그래픽 렌더링 API =================
void graphics_draw_rectangle(FrameBuffer* frame, int x, int y, int w, int h, int thickness, unsigned int color);
void graphics_draw_text(FrameBuffer* frame, const char* text, int x, int y, int font_size, unsigned int color);

// ================= 4. LCD 디스플레이 API =================
int lcd_display_frame(const FrameBuffer* frame);

// ================= 5. 저장 장치 API =================
int storage_start_recording(const char* filename);
void storage_stop_recording();
int storage_write_frame(const FrameBuffer* frame);

// ================= 6. CAN 통신 API =================
typedef struct {
    unsigned int id;
    unsigned char dlc;
    unsigned char data[8];
} CANMessage;

int can_init(int bitrate);
int can_send_message(const CANMessage* msg);
int can_receive_message(CANMessage* msg); // 1=수신, 0=없음, <0=에러
void can_close();

#ifdef __cplusplus
}
#endif
#endif // HARDWARE_H