#include <Arduino.h>
#include <SPI.h>
#include <mcp_can.h>

// ====== MCP2515 (UNO) ======
#define CAN_CS_PIN   10
#define CAN_INT_PIN  2      // (옵션) 사용 안해도 됨
#define CAN_BITRATE  CAN_500KBPS
#define CAN_CLOCK    MCP_8MHZ   // ★ 모듈 크리스탈 8MHz (16MHz면 MCP_16MHZ)

MCP_CAN CAN0(CAN_CS_PIN);

// ====== 시리얼 프레임 정의 ======
const uint8_t STX0 = 0xAA;
const uint8_t STX1 = 0x55;
const uint8_t MSG_VEH_STATUS  = 0x01; // 8B payload: <HhBBBB>
const uint8_t MSG_VEH_STATUS2 = 0x02; // 10B payload: <id16><HhBBBB>

enum ParseState { WAIT_STX0, WAIT_STX1, WAIT_MSGID, WAIT_LEN, READ_PAYLOAD, WAIT_CRC };
ParseState state = WAIT_STX0;
uint8_t msg_id=0, len=0, idx=0, crc=0;
const uint8_t MAX_PAYLOAD=24;
uint8_t payload[MAX_PAYLOAD];

uint8_t crc8_xor_update(uint8_t c, uint8_t b){ return c ^ b; }
void reset_parser(){ state=WAIT_STX0; msg_id=0; len=0; idx=0; crc=0; }

// ====== 텍스트 라인 버퍼/파서 ======
const size_t LINE_MAX = 96;
char linebuf[LINE_MAX];
size_t lineidx = 0;

// 텍스트 한 줄을 <HhBBBB>로 변환
bool parse_text_line(const char* s, uint8_t out8[8]) {
  // 예: SPD=  7.24 kph  STR=  0.15 deg  THR= 23%  BRK=  0%  G=1  FLG=0x00
  float spd = 0.f, steer_deg = 0.f;
  int thr=0, brk=0, gear=0;
  unsigned flags = 0;

  // 공백은 유동적 → 단위 토큰은 *로 스킵
  int n = sscanf(s,
    "SPD=%f %*s STR=%f %*s THR=%d%% BRK=%d%% G=%d FLG=0x%x",
    &spd, &steer_deg, &thr, &brk, &gear, &flags);

  if (n != 6) return false;

  long speed01 = lroundf(spd * 10.0f);       // 0.1 kph → uint16
  long steer01 = lroundf(steer_deg * 10.0f); // 0.1 deg → int16
  if (speed01 < 0) speed01 = 0;
  if (speed01 > 65535) speed01 = 65535;
  if (steer01 < -32768) steer01 = -32768;
  if (steer01 >  32767) steer01 =  32767;
  if (thr < 0) thr = 0; if (thr > 100) thr = 100;
  if (brk < 0) brk = 0; if (brk > 100) brk = 100;
  if (gear < 0) gear = 0; if (gear > 255) gear = 255;
  if (flags > 255) flags = 255;

  // <HhBBBB> (LE)
  out8[0] = (uint8_t)(speed01 & 0xFF);
  out8[1] = (uint8_t)((speed01 >> 8) & 0xFF);
  out8[2] = (uint8_t)((int16_t)steer01 & 0xFF);
  out8[3] = (uint8_t)(((int16_t)steer01 >> 8) & 0xFF);
  out8[4] = (uint8_t)thr;
  out8[5] = (uint8_t)brk;
  out8[6] = (uint8_t)gear;
  out8[7] = (uint8_t)flags;
  return true;
}

// ====== 디버그 출력 ======
void print_v1(const uint8_t* p){
  uint16_t speed01 = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
  int16_t  steer01 = (int16_t)((uint16_t)p[2] | ((uint16_t)p[3]<<8));
  uint8_t thr=p[4], brk=p[5], gear=p[6], flags=p[7];
  Serial.print(F("[HERO] SPD=")); Serial.print(speed01/10.0f,1);
  Serial.print(F(" kph STR=")); Serial.print(steer01/10.0f,1);
  Serial.print(F(" THR=")); Serial.print(thr);
  Serial.print(F("% BRK=")); Serial.print(brk);
  Serial.print(F("% G=")); Serial.print(gear);
  Serial.print(F(" AP=")); Serial.println((flags&1)?F("ON"):F("OFF"));
}

void print_v2(const uint8_t* p){
  uint16_t id16    = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
  uint16_t speed01 = (uint16_t)p[2] | ((uint16_t)p[3]<<8);
  int16_t  steer01 = (int16_t)((uint16_t)p[4] | ((uint16_t)p[5]<<8));
  uint8_t thr=p[6], brk=p[7], gear=p[8], flags=p[9];
  Serial.print(F("[V")); Serial.print(id16); Serial.print(F("] "));
  Serial.print(F("SPD=")); Serial.print(speed01/10.0f,1);
  Serial.print(F(" kph STR=")); Serial.print(steer01/10.0f,1);
  Serial.print(F(" THR=")); Serial.print(thr);
  Serial.print(F("% BRK=")); Serial.print(brk);
  Serial.print(F("% G=")); Serial.print(gear);
  Serial.print(F(" AP=")); Serial.println((flags&1)?F("ON"):F("OFF"));
}

// ====== CAN 송신 래퍼 ======
void send_can_payload_v1(const uint8_t data8[8]) {
  byte stat = CAN0.sendMsgBuf(0x123, 0, 8, (byte*)data8);
  if (stat != CAN_OK) {
    Serial.println(F("[CAN] TX fail (V1)"));
  }
}

void setup(){
  Serial.begin(115200);
  delay(500);
  pinMode(LED_BUILTIN, OUTPUT);

  if (CAN0.begin(MCP_ANY, CAN_BITRATE, CAN_CLOCK) != CAN_OK) {
    Serial.println(F("[ERR] MCP2515 init failed"));
    while (1) { delay(100); }
  }
  CAN0.setMode(MCP_NORMAL);
  Serial.println(F("[OK] MCP2515 NORMAL 500kbps @8MHz"));
  Serial.println(F("UNO RX ready (text & binary) -> CAN TX"));
}

void loop(){
  while (Serial.available() > 0) {
    uint8_t b = (uint8_t)Serial.read();

    // --- 1) 텍스트 라인 수집 (개행까지 모음) ---
    if (b == '\n' || b == '\r') {
      if (lineidx > 0) {
        linebuf[lineidx] = '\0';
        uint8_t data8[8];
        if (parse_text_line(linebuf, data8)) {
          digitalWrite(LED_BUILTIN, HIGH);
          send_can_payload_v1(data8);
          print_v1(data8);                 // 디버그
          digitalWrite(LED_BUILTIN, LOW);
        }
        lineidx = 0;
      }
      // 개행 처리 끝나면 다음 바이트로
      // (바이너리 파서는 개행을 쓰지 않으니 여기서 continue 안 해도 무방)
    } else {
      if (lineidx < LINE_MAX - 1) {
        linebuf[lineidx++] = (char)b;
      } else {
        // 라인 오버플로 → 리셋
        lineidx = 0;
      }
    }

    // --- 2) 바이너리 프레임 파서 (AA 55 ... CRC) ---
    switch (state) {
      case WAIT_STX0:
        if (b == STX0) { crc = 0; crc = crc8_xor_update(crc, b); state = WAIT_STX1; }
        break;

      case WAIT_STX1:
        if (b == STX1) { crc = crc8_xor_update(crc, b); state = WAIT_MSGID; }
        else state = WAIT_STX0;
        break;

      case WAIT_MSGID:
        msg_id = b; crc = crc8_xor_update(crc, b); state = WAIT_LEN; break;

      case WAIT_LEN:
        len = b; crc = crc8_xor_update(crc, b);
        if (len > MAX_PAYLOAD) { reset_parser(); }
        else { idx = 0; state = READ_PAYLOAD; }
        break;

      case READ_PAYLOAD:
        payload[idx++] = b; crc = crc8_xor_update(crc, b);
        if (idx >= len) state = WAIT_CRC;
        break;

      case WAIT_CRC:
        if (b == crc) {
          digitalWrite(LED_BUILTIN, HIGH);

          if (msg_id == MSG_VEH_STATUS && len == 8) {
            // V1: 8B -> CAN 0x123
            byte stat = CAN0.sendMsgBuf(0x123, 0, 8, payload);
            if (stat == CAN_OK) print_v1(payload);
            else Serial.println(F("[CAN] TX fail (V1)"));

          } else if (msg_id == MSG_VEH_STATUS2 && len == 10) {
            // V2: id16(2B) + 8B -> CAN (0x500 | id16&0x7FF)
            uint16_t id16 = (uint16_t)payload[0] | ((uint16_t)payload[1] << 8);
            uint8_t data8[8];
            for (int i=0;i<8;i++) data8[i] = payload[2+i];
            uint16_t can_id = 0x500 | (id16 & 0x7FF);
            byte stat = CAN0.sendMsgBuf(can_id, 0, 8, data8);
            if (stat == CAN_OK) { Serial.println(F("Message Sent OK!")); print_v2(payload); }
            else Serial.println(F("[CAN] TX fail (V2)"));

          } else {
            Serial.println(F("[WARN] Unknown msg/len"));
          }

          digitalWrite(LED_BUILTIN, LOW);
        } else {
          Serial.print(F("[ERR] CRC mismatch got=0x"));
          if (b < 16) Serial.print('0'); Serial.print(b, HEX);
          Serial.print(F(" exp=0x")); if (crc < 16) Serial.print('0'); Serial.println(crc, HEX);
        }
        reset_parser();
        break;
    }
  }
}
