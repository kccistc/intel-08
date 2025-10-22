#ifndef HARDWARE_H
#define HARDWARE_H
#include <stddef.h>
#include <time.h>
#ifdef __cplusplus
extern "C" {
#endif
// ================= 1. 공통 및 초기화 API =================

// --- CAN 통신 관련 변수 ---
#define PID_ENGINE_SPEED            0x0c //업계 표준, RPM 
#define PID_VEHICLE_SPEED           0x0d //업계 표준, 속도
#define PID_GEAR_STATE              0xa4 //업계 표준(특정 차에서는 안될수도 있음)
#define PID_GPS_XDATA               0x10 //GPS 데이터(실제 존재 x)
#define PID_GPS_YDATA               0x11 //GPS 데이터(실제 존재 x)
#define PID_STEERING_DATA           0x20 //조향각 데이터(실제 존재 X)
#define PID_BRAKE_DATA              0x40 //브레이크 데이터(실제 존재 X)
#define PID_TIRE_DATA               0x80 //타이어 공기압 데이터(실제 존재 X)
#define PID_THROTTLE_DATA           0x50 //스로틀 데이터(실제 존재 X)

// --- 상태 제어 관련 변수 ---
#define ENGINE_SPEED_FLAG           0x01
#define VEHICLE_SPEED_FLAG          0x02
#define GEAR_STATE_FLAG             0x04
#define GPS_XDATA_FLAG              0x08
#define GPS_YDATA_FLAG              0x10
#define STEERING_DATA_FLAG          0x20
#define BRAKE_DATA_FLAG             0x40
#define TIRE_DATA_FLAG              0x80

#define THROTTLE_DATA_FLAG          0x01

// --- 위험 상태 관련 변수 ---
#define MAX_DISTANCE                2.0

// --- 가속도 측정 관련 변수 ---
#define SPEED_BUF                   32 // 최근 32개 사이클 저장
#define KPH_TO_MPS                  (1.0/3.6)

// 튜닝 임계값(조절하면서 튜닝)
#define ACCEL_THRESH_MPS2           2.5 // 급가속: +2.5 m/s^2 이상
#define DECEL_THRESH_MPS2           (-3.0) // 급감속: -3.0 m/s^2 이하
#define DV10_KPH_THRESH             15.0 // 10사이클 전 대비 15 km/h 이상 변화

#define AI_REQUEST_FLAG             0x01
#define AI_RESULT_READY_FLAG        0x02
#define AI_RESEULT_ERROR_FLAG        0x04

#define GPS_AVAILABLE               (GPS_XDATA_FLAG|GPS_YDATA_FLAG)
#define AI_AVAILABLE                (GPS_XDATA_FLAG|GPS_YDATA_FLAG|STEERING_DATA_FLAG)
#define COMPLETE_DATA_FLAG          (ENGINE_SPEED_FLAG|VEHICLE_SPEED_FLAG|GEAR_STATE_FLAG|GPS_XDATA_FLAG|GPS_YDATA_FLAG|STEERING_DATA_FLAG|BRAKE_DATA_FLAG|TIRE_DATA_FLAG)
#define DATA_AVAILABLE              (COMPLETE_DATA_FLAG & (~AI_AVAILABLE))

#define ACCELRATION                 0x01 //급가속 감지                 
#define DECELERATION                0x02 //급감속 감지
#define DETECT_HUMAN                0x04 //사람 감지
#define DETECT_TRUCK                0x10 // 트럭 감지
#define DETECT_ODOBANGS             0x20 // 오토바이, 자전거 감지
#define DETECT_FUNK                 0x40 // 펑크 감지

#define TIRE_PRESSURE_THRESHOLD    30 // 타이어 공기압 임계값(psi)

//라벨
#define LABEL_CAR                   0
#define LABEL_TRUCK                 1
#define LABEL_CONSTRUCTION_TRUCK    2
#define LABEL_BUS                   3
#define LABEL_TRAILER               4
#define LABEL_BARRIER               5
#define LABEL_MOTOCYCLE             6
#define LABEL_BICYCLE               7
#define LABEL_PEDESTRIAN            8
#define LABEL_TRAFFIC_CONE          9

//AI 요청 프레임 속도
#define FPS_TARGET                  5.0

typedef struct {
    double v_kph[SPEED_BUF];
    double t_sec[SPEED_BUF];
    int    idx;
    int    count;
} SpeedMonitor;

// 현재 시간 측정
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// 히스토리에 (속도[km/h], 시간[s]) 추가
static void spmon_push(SpeedMonitor *m, double v_kph, double t_sec) {
    m->v_kph[m->idx] = v_kph;
    m->t_sec[m->idx] = t_sec;
    m->idx = (m->idx + 1) % SPEED_BUF;
    if (m->count < SPEED_BUF) m->count++;
}

// i번째 과거 샘플 읽기
static int spmon_get_past(const SpeedMonitor *m, int offset, double *v_kph, double *t_sec) {
    if (offset >= m->count) return -1;
    int pos = (m->idx - 1 - offset + SPEED_BUF) % SPEED_BUF;
    if (v_kph) *v_kph = m->v_kph[pos];
    if (t_sec) *t_sec = m->t_sec[pos];
    return 0;
}

//==================CAN통신 관련====================

typedef struct{
    unsigned char pid; // 요청할 PID
    unsigned char flag; //PID 응답 저장 플래그
}CANRequest;

// ================= 2. 카메라 API =================
typedef struct {
    unsigned char* data; // RGB24
    int width;
    int height;
    size_t size;          // bytes
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

typedef struct {
    double gps_x;
    double gps_y;
    int speed;
    int rpm;
    unsigned char brake_state;
    float gear_ratio;
    char gear_state;
    float degree;
    unsigned char throttle;
    unsigned char tire_pressure[4];
} VehicleData;

int can_request_pid(unsigned char pid);
void can_parse_and_update_data(const CANMessage* msg, VehicleData* vehicle_data, unsigned char* flag, unsigned char* flag2);
int can_init(const char* interface_name); // <<-- 수정: 인터페이스 이름을 받고, 성공 시 fd를 반환하도록 변경
int can_send_message(const CANMessage* msg);
int can_receive_message(CANMessage* msg); // 1=수신, 0=없음, <0=에러
void can_close();

// ================= 7. AI 통신 API =================
typedef struct{
    unsigned char label;
    float x;
    float y;
    float ax;
    float ay;
}DetectedObject;

#ifdef __cplusplus
}
#endif
#endif
