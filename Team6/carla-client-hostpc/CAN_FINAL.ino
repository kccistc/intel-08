/*
 * Arduino ECU Emulator (MCP2515, 8MHz, 500 kbps) — REQ/RES 시점 상태 로그 (퍼센트 응답형)
 *  - 0x7DF 요청 수신 시 [REQ] 라인 출력 (PID/원본데이터 + 상태 요약)
 *  - 응답 전송 시 [RES] 라인 출력 (PID/페이로드 HEX + 상태 요약)
 *  - BRAKE(0x40), THROTTLE(0x50) 모두 0~100% '값'으로 요청-응답
 */

#include <SPI.h>
#include <mcp_can.h>
#include <math.h>
#include <string.h>    // memcpy

/* ===================== 로깅 설정 ===================== */
#define LOG_REQRES        1   // [REQ]/[RES] 라인 출력
#define LOG_PAYLOAD_HEX   1   // [RES]에서 페이로드 HEX 같이 출력
#define REQ_THROTTLE_MS  50   // 같은 PID 반복요청 로그 최소 간격(ms)

/* ===================== 하드웨어/통신 설정 ===================== */
#define CAN_CS_PIN    10
#define CAN_INT_PIN    2
#define CAN_CLOCK     MCP_8MHZ
#define CAN_SPEED     CAN_500KBPS
#define CAN_MODE      MCP_ANY
#define CAN_ID_REQ    0x7DF
#define CAN_ID_RES    0x7E8

/* ===================== PID 정의 ===================== */
#define PID_RPM   0x0C
#define PID_SPEED 0x0D
#define PID_GEAR  0xA4
#define PID_GPS_X 0x10
#define PID_GPS_Y 0x11
#define PID_STEER 0x20
#define PID_BRAKE 0x40   // 0~100%
#define PID_THOTL 0x50   // 0~100%
#define PID_TIRE  0x80

/* ===================== 시리얼 프레임 포맷 =====================
 * [AA][55][90][LEN][PID][VALUE...][CRC]  (LEN = PID+VALUE 길이)
 * CRC = XOR([AA 55 90 LEN]) ^ XOR([PID VALUE...])
 */
#define SER_STX0 0xAA
#define SER_STX1 0x55
#define SER_MSG_PID_UPDATE 0x90

/* ===================== 내부 상태/출력 정책 ===================== */
static unsigned long g_last_req_log_ms = 0;
static uint8_t       g_last_req_pid   = 0xFF;
unsigned long long cnt = 0;
volatile uint8_t   g_speed = 0;            // km/h
volatile uint16_t  g_rpm_x4 = 0;           // rpm*4 (OBD 포맷)
volatile uint16_t  g_ratio_x1000 = 0;      // gear ratio*1000
volatile uint8_t   g_gear_code = 0;        // 상위 4bit 사용
volatile float     g_x_m = 0.0f;           // GPS X (m)
volatile float     g_y_m = 0.0f;           // GPS Y (m)
volatile uint8_t   g_steer_S = 1;          // 1:+, 0:-
volatile uint8_t   g_steer_I = 0;          // 정수부
volatile uint8_t   g_steer_F = 0;          // 소수부(센티도)
volatile uint8_t   g_brake = 0;            // 0~100 (%)
volatile uint8_t   g_throttle = 0;         // 0~100 (%)
volatile uint8_t   g_tp[4] = {40,40,40,40}; // tire kPa 예시

MCP_CAN CAN0(CAN_CS_PIN);

/* ===================== 유틸 ===================== */
static inline uint8_t clamp_0_99(int v){ if(v<0) return 0; if(v>99) return 99; return (uint8_t)v; }
static inline uint8_t clamp_0_100(int v){ if(v<0) return 0; if(v>100) return 100; return (uint8_t)v; }

static inline float steer_deg_value(){
  float mag = (float)g_steer_I + (float)g_steer_F/100.0f;
  return (g_steer_S ? +1.0f : -1.0f) * mag;
}

static void print_hex(const uint8_t* p, uint8_t n){
  for(uint8_t i=0;i<n;i++){
    if(i) Serial.print(' ');
    if(p[i]<0x10) Serial.print('0');
    Serial.print(p[i], HEX);
  }
}

uint8_t crc8_xor(const uint8_t* p, uint16_t n){
  uint8_t c=0; for(uint16_t i=0;i<n;++i) c^=p[i]; return c;
}

static const __FlashStringHelper* pid_name(uint8_t pid){
  switch(pid){
    case PID_SPEED: return F("SPEED");
    case PID_RPM:   return F("RPM");
    case PID_GEAR:  return F("GEAR");
    case PID_GPS_X: return F("GPS_X");
    case PID_GPS_Y: return F("GPS_Y");
    case PID_STEER: return F("STEER");
    case PID_BRAKE: return F("BRAKE");
    case PID_THOTL: return F("THROTTLE");
    case PID_TIRE:  return F("TIRE");
    default:        return F("UNKNOWN");
  }
}

/* 공통 상태 요약 한 줄 */
static void print_state_summary(){
  Serial.print(F("  | X="));   Serial.print(g_x_m, 6);
  Serial.print(F(" Y="));       Serial.print(g_y_m, 6);
  Serial.print(F(" SPD="));     Serial.print(g_speed);
  Serial.print(F(" RPM="));     Serial.print((uint16_t)(g_rpm_x4/4));
  Serial.print(F(" STEER="));   Serial.print(steer_deg_value(), 2);
  Serial.print(F(" BRK="));     Serial.print(g_brake);    Serial.print('%');
  Serial.print(F(" THR="));     Serial.print(g_throttle); Serial.print('%');
  Serial.print(F(" TIRE="));    Serial.print(g_tp[0]); Serial.print('/');
  Serial.print(g_tp[1]); Serial.print('/'); Serial.print(g_tp[2]); Serial.print('/');
  Serial.print(g_tp[3]);
}

static void print_state_line_manual(){
  Serial.print(F("[STATE]"));
  print_state_summary();
  Serial.println();
}

/* ===================== 시리얼 수신 FSM ===================== */
enum RX_STATE { ST_WAIT_AA, ST_WAIT_55, ST_WAIT_ID, ST_WAIT_LEN, ST_WAIT_BODY, ST_WAIT_CRC };
RX_STATE rx_state = ST_WAIT_AA;
uint8_t  rx_id=0, rx_len=0, rx_buf[32], rx_pos=0;

void process_pid_update(const uint8_t* body, uint8_t len){
  if (len < 1) return;
  uint8_t pid = body[0];
  const uint8_t* payload = body + 1;
  uint8_t plen = (len >= 1) ? (len - 1) : 0;

  switch(pid){
    case PID_SPEED:
      if(plen>=1){ g_speed = payload[0]; }
      break;

    case PID_RPM:
      if(plen>=2){ g_rpm_x4 = ((uint16_t)payload[0]<<8) | payload[1]; }
      break;

    case PID_GEAR:
      if(plen>=3){
        g_ratio_x1000 = ((uint16_t)payload[0]<<8) | payload[1];
        g_gear_code   = (payload[2]>>4) & 0x0F;
      }
      break;

    case PID_GPS_X:
      if(plen>=4){ memcpy((void*)&g_x_m, payload, 4); }
      break;
    case PID_GPS_Y:
      if(plen>=4){ memcpy((void*)&g_y_m, payload, 4); }
      break;

    case PID_STEER:
      if(plen>=3){
        g_steer_S = payload[0] ? 1 : 0;
        g_steer_I = payload[1];
        g_steer_F = clamp_0_99(payload[2]);
      } else if(plen>=2){
        g_steer_S = 1;
        g_steer_I = payload[0];
        g_steer_F = clamp_0_99(payload[1]);
      }
      break;

    case PID_BRAKE:                 // 브레이크 0~100%
      if(plen>=1){ g_brake = clamp_0_100(payload[0]); }
      break;

    case PID_THOTL:                 // 스로틀 0~100%
      if(plen>=1){ g_throttle = clamp_0_100(payload[0]); }
      break;

    case PID_TIRE:
      if(plen>=4){ g_tp[0]=payload[0]; g_tp[1]=payload[1]; g_tp[2]=payload[2]; g_tp[3]=payload[3]; }
      break;

    default: break;
  }
}

void serial_rx_fsm(){
  while(Serial.available()){
    uint8_t b = (uint8_t)Serial.read();
    switch(rx_state){
      case ST_WAIT_AA: if(b==SER_STX0) rx_state=ST_WAIT_55; break;
      case ST_WAIT_55: rx_state=(b==SER_STX1)?ST_WAIT_ID:ST_WAIT_AA; break;
      case ST_WAIT_ID: rx_id=b; rx_state=ST_WAIT_LEN; break;
      case ST_WAIT_LEN:
        rx_len=b;
        if(rx_len>sizeof(rx_buf)){ rx_state=ST_WAIT_AA; break; }
        rx_pos=0; rx_state=(rx_len>0)?ST_WAIT_BODY:ST_WAIT_CRC; break;
      case ST_WAIT_BODY:
        rx_buf[rx_pos++]=b; if(rx_pos>=rx_len) rx_state=ST_WAIT_CRC; break;
      case ST_WAIT_CRC:{
        uint8_t head[4]={SER_STX0,SER_STX1,rx_id,rx_len};
        uint8_t c = crc8_xor(head,4) ^ crc8_xor(rx_buf,rx_len);
        if(c==b && rx_id==SER_MSG_PID_UPDATE){ process_pid_update(rx_buf, rx_len); }
        rx_state=ST_WAIT_AA; break;
      }
    }
  }
}

/* ===================== 공통 OBD 응답 조립/송신 ===================== */
void send_obd_response(uint8_t pid, const uint8_t* payload, uint8_t plen){
  uint8_t d[8]={0};
  uint8_t L = 2 + plen; if(L>7) L=7;   // 단일 프레임 한계
  d[0]=L; d[1]=0x41; d[2]=pid;
  for(uint8_t i=0;i<plen && (3+i)<8;i++) d[3+i]=payload[i];
  CAN0.sendMsgBuf(CAN_ID_RES, 0, 8, d);

#if LOG_REQRES
  Serial.print(F("[RES] PID=0x"));
  if(pid<0x10) Serial.print('0');
  Serial.print(pid, HEX);
  Serial.print(F(" (")); Serial.print(pid_name(pid)); Serial.print(F(")"));
#if LOG_PAYLOAD_HEX
  Serial.print(F("  payload:"));
  print_hex(payload, (plen>5?5:plen));
#endif
  print_state_summary();
  Serial.println();
#endif
}

/* ===================== GPS S/I/D2/D4/D6 인코딩 ===================== */
void encode_SI_D2D4D6(float meters, uint8_t out[5]){
  float a = fabsf(meters);
  uint8_t S = (meters >= 0.0f) ? 1 : 0;
  uint8_t I = (uint8_t)(a >= 255.0f ? 255 : (uint8_t)floorf(a));
  float frac = a - (float)I;
  uint32_t frac6 = (uint32_t) lroundf(frac*1000000.0f);
  if(frac6>999999) frac6=999999;
  uint8_t D2 = clamp_0_99((int)((frac6/10000)%100));
  uint8_t D4 = clamp_0_99((int)((frac6/  100)%100));
  uint8_t D6 = clamp_0_99((int)( frac6       %100));
  out[0]=S; out[1]=I; out[2]=D2; out[3]=D4; out[4]=D6;
}

/* ===================== PID별 응답 함수 ===================== */
void respond_speed(){ uint8_t p[1]={g_speed}; send_obd_response(PID_SPEED,p,1); }
void respond_rpm(){ uint8_t p[2]={(uint8_t)(g_rpm_x4>>8),(uint8_t)g_rpm_x4}; send_obd_response(PID_RPM,p,2); }
void respond_gear(){ uint8_t p[3]={(uint8_t)(g_ratio_x1000>>8),(uint8_t)g_ratio_x1000,(uint8_t)((g_gear_code&0x0F)<<4)}; send_obd_response(PID_GEAR,p,3); }
void respond_gps_x(){ uint8_t p[5]; encode_SI_D2D4D6(g_x_m,p); send_obd_response(PID_GPS_X,p,5); }
void respond_gps_y(){ uint8_t p[5]; encode_SI_D2D4D6(g_y_m,p); send_obd_response(PID_GPS_Y,p,5); }
void respond_steer(){ uint8_t p[3]={ g_steer_S?1:0, g_steer_I, g_steer_F }; send_obd_response(PID_STEER,p,3); }
void respond_brake(){ uint8_t p[1]={ g_brake };     send_obd_response(PID_BRAKE,p,1); }    // 0~100%
void respond_throttle(){ uint8_t p[1]={ g_throttle }; send_obd_response(PID_THOTL,p,1); }  // 0~100%
//void respond_tire(){ uint8_t p[4]={ g_tp[0],g_tp[1],g_tp[2],g_tp[3] }; send_obd_response(PID_TIRE,p,4); }
void respond_tire() {
  uint8_t p[4];

  // cnt가 10000 이상이면 항상 40/39/40/20으로 응답
  if (cnt == 10000ULL) {
    p[0] = 40;
    p[1] = 39;
    p[2] = 40;
    p[3] = 20;

    cnt = 0 ;
  } else {
    p[0] = g_tp[0];
    p[1] = g_tp[1];
    p[2] = g_tp[2];
    p[3] = g_tp[3];
  }

  send_obd_response(PID_TIRE, p, 4);
}

/* ===================== CAN 요청 처리 ===================== */
void handle_can_request(const uint8_t* buf, uint8_t len){
  if(len<3) return;
  if(buf[0]<0x02) return;
  if(buf[1]!=0x01) return;

  uint8_t pid = buf[2];

#if LOG_REQRES
  // 같은 PID 연속 로그 과다 방지
  unsigned long now = millis();
  if (pid != g_last_req_pid || (now - g_last_req_log_ms) >= REQ_THROTTLE_MS){
    Serial.print(F("[REQ] PID=0x"));
    if(pid<0x10) Serial.print('0');
    Serial.print(pid, HEX);
    Serial.print(F(" (")); Serial.print(pid_name(pid)); Serial.print(F(")  raw:"));
    print_hex(buf, len);
    print_state_summary();
    Serial.println();
    g_last_req_pid = pid;
    g_last_req_log_ms = now;
  }
#endif

  switch(pid){
    case PID_SPEED: respond_speed(); break;
    case PID_RPM:   respond_rpm();   break;
    case PID_GEAR:  respond_gear();  break;
    case PID_GPS_X: respond_gps_x(); break;
    case PID_GPS_Y: respond_gps_y(); break;
    case PID_STEER: respond_steer(); break;
    case PID_BRAKE: respond_brake(); break;     // 퍼센트 값 그대로
    case PID_THOTL: respond_throttle(); break;  // 퍼센트 값 그대로
    case PID_TIRE:  respond_tire();  break;
    default: break;
  }
}

/* ===================== Arduino Setup/Loop ===================== */
void setup(){
  Serial.begin(115200);
  while(CAN_OK != CAN0.begin(CAN_MODE, CAN_SPEED, CAN_CLOCK)){ delay(100); }
  pinMode(CAN_INT_PIN, INPUT);
  CAN0.setMode(MCP_NORMAL);
  Serial.println(F("MCP2515 OK"));
  print_state_line_manual(); // 부팅 직후 1회 상태 출력
}


void loop(){
  serial_rx_fsm();
  cnt++;

  if (CAN_MSGAVAIL == CAN0.checkReceive()){
    long unsigned id; unsigned char len; unsigned char buf[8];
    CAN0.readMsgBuf(&id,&len,buf);
    if (id == CAN_ID_REQ){
      handle_can_request(buf,len);
    }
  }
}
