# V2X Accident Alert over Wi-Fi (UDP Multicast)

서버(PC/RSU)가 **사고 알림**을 멀티캐스트로 방송하면, 차량(Jetson/Raspberry Pi) 클라이언트가 수신하여 터미널 출력/CSV 로깅/ROS2 퍼블리시까지 수행합니다.

## 파일
- `server.py` — **서버(송신)**: 사고 알림 브로드캐스트
- `client.py` — **차량(수신)**: 알림 수신 + (선택) ROS2 퍼블리시

## 요구 사항
- Python 3.8+
- 같은 Wi-Fi AP(5 GHz 권장)에 서버/차량이 모두 연결
- (선택) ROS2 Humble 이상(차량에서 `--ros2` 사용 시)

---

### 1) 차량(수신) 먼저 실행
python3 v2x_alert_client.py --log alerts.csv
NIC가 여러 개면 예) --iface 192.168.0.23

### 2) 서버(송신)에서 반복 브로드캐스트
python3 v2x_alert_server.py --repeat \
  --type collision --severity high --distance-m 500 \
  --suggest slow_down --road segment_A \
  --lat 37.12345 --lon 127.12345

### 3) (선택) 무결성 검증(HMAC)

두쪽에 같은 키를 지정하세요:

export V2X_KEY="mysecret" \
python3 v2x_alert_server.py --repeat --hmac-key "$V2X_KEY" \
python3 v2x_alert_client.py --hmac-key "$V2X_KEY" 

### 4) (선택) ROS2 퍼블리시(차량)
python3 v2x_alert_client.py --ros2 \
토픽: /v2x/alert (std_msgs/String, payload는 JSON 문자열)

### 메시지 포맷(JSON)
{
  "hdr": {"ver":1,"src":"rsu_server_01","seq":42,"ts":1690000000.0}, \
  "accident": {  \
    "type":"collision",        // collision | fire | rollover | blockage | unknown \
    "severity":"high",         // low | medium | high | critical \
    "distance_m":500.0, \
    "road":"segment_A", \
    "lat":37.12345, "lon":127.12345 \
  }, \
  "advice": {"message":"", "suggest":"slow_down"}, // slow_down | stop | detour | caution \
  "ttl_s":10.0,                                     // 메시지 유효시간(초) \
  "sig": {"alg":"hmac-sha256","value":"..."}        // (HMAC 사용 시) \
}


클라이언트는 hdr.ts로부터 ttl_s를 초과한 메시지를 자동 폐기합니다.

서버는 --repeat --hz 2.0 등으로 반복 송신하여 누락 대비 권장.

### 서버 옵션(v2x_alert_server.py)
--mcast        멀티캐스트 그룹 (기본: 239.20.20.20) \
--port         포트 (기본: 5520) \
--iface        송신 인터페이스 IPv4(옵션) \
--repeat       반복 송신 (미지정 시 3회 원샷) \
--hz           반복 주기 Hz (기본: 2.0) \
--src-id       헤더 소스 ID (기본: rsu_server_01) \
--hmac-key     HMAC 키(무결성용, 옵션) \
--ttl-s        메시지 유효시간 초 (기본: 10.0) 

--type         사고 유형 (collision|fire|rollover|blockage|unknown) \
--severity     심각도 (low|medium|high|critical) \
--distance-m   차량군 기준 거리(미터) \
--road         도로/세그먼트 이름 \
--lat --lon    사고 위치 좌표 \
--suggest      조치 권고 (slow_down|stop|detour|caution) \
--message      자유 텍스트 메시지 

### 클라이언트 옵션(v2x_alert_client.py)
--mcast        멀티캐스트 그룹 (기본: 239.20.20.20) \
--port         포트 (기본: 5520) \
--iface        가입 인터페이스 IPv4(옵션) \
--hmac-key     수신 HMAC 검증 키(옵션) \
--log          CSV 로그 파일 경로(옵션) \
--ros2         ROS2 토픽(/v2x/alert) 퍼블리시 활성화 


### CSV 컬럼:

recv_ts,seq,src,type,severity,distance_m,road,lat,lon,suggest,ok_hmac


### 네트워크 팁 & 문제해결

같은 서브넷/같은 AP에 연결하세요. 5 GHz 권장.

일부 AP는 멀티캐스트를 차단/제한 → 유선 스위치/라우터 직결 테스트 추천.

인터페이스가 여러 개면 --iface 명시.

패킷 캡처 확인:

sudo tcpdump -i wlan0 udp port 5520


