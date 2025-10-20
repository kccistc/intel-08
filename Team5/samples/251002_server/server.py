import paho.mqtt.client as mqtt
import pymysql
from datetime import datetime, timezone

# === DB 연결 (MariaDB) ===
db = pymysql.connect(
    host="localhost",
    user="marine_user",       # 위에서 만든 사용자
    password="sksk",          # 설정한 비밀번호
    database="marine",
    charset="utf8mb4"
)
cursor = db.cursor()

# === MQTT 설정 ===
BROKER = "0.0.0.0"
PORT = 1883
TOPIC = "project/#"

# === 콜백 ===
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[OK] Connected to broker")
        client.subscribe(TOPIC)
        print(f"[SUB] Subscribed to {TOPIC}")
    else:
        print("[FAIL] Connection failed, code:", rc)

def on_message(client, userdata, msg):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") # UTC 기준
    payload = msg.payload.decode()
    topic = msg.topic
    print(f"[{now}] {topic} → {payload}")

    # 로그 DB 저장
    try:
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        # topic은 보통 "project/모듈명" 이므로 split
        module = topic.split("/")[1] if "/" in topic else topic
        action = "EVT"   # 기본값, 필요하면 파싱해서 변경 가능
        cursor.execute(sql, (module, action, payload, now))
        db.commit()
    except Exception as e:
        print(f"[DB-ERROR] {e}")
# === MQTT 클라이언트 생성 ===
client = mqtt.Client(client_id="MarineServer")
client.on_connect = on_connect
client.on_message = on_message

# === 브로커 연결 ===
print("[INFO] Connecting to broker...")
client.connect(BROKER, PORT, 60)

# === 루프 ===
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[EXIT] Server stopped by user")
    client.disconnect()
    cursor.close()
    db.close()

