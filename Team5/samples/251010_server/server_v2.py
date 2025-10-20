import paho.mqtt.client as mqtt
import pymysql
from datetime import datetime, timezone
from gtts import gTTS
import os
import requests # Hugging Face API 용

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

# === Hugging Face LLM 설정 ===
HF_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# === LLM 질의응답 함수 ===
def query_llm(prompt):
    payload = {"inputs": prompt}
    try:
        res = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            else:
                return str(data)
        elif res.status_code == 403:
            print("[LLM Error] Access forbidden (403) — check token or model permissions.")
            return "⚠️ LLM 접근이 제한되었습니다."
        else:
            print(f"[LLM Error] HTTP {res.status_code}: {res.text[:200]}")
            return f"⚠️ LLM 오류 ({res.status_code})"
    except Exception as e:
        print(f"[LLM Error] {e}")
        return "⚠️ LLM 요청 중 오류 발생."

# === 로그 불러오기 ===
def fetch_logs(minutes=10):
    try:
        sql = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        cursor.execute(sql, (minutes,))
        rows = cursor.fetchall()
        if not rows:
            return [f"최근 {minutes}분 동안 이벤트가 없습니다."]
        logs = [f"[{r[0]}] ({r[1]}) {r[2]} → {r[3]}" for r in rows]
        print(f"[DB] Retrieved {len(logs)} logs for summary")
        return logs
    except Exception as e:
        print(f"[DB-ERROR] fetch_logs: {e}")
        return ["로그 불러오기 실패."]
    
# === LLM 요약 ===
def summarize_logs(logs):
    text = "\n".join(logs)
    prompt = f"""
    다음은 선박 항해 로그입니다:
    {text}

    위 로그를 간결하고 구조적으로 요약해줘. (한국어로)
    """
    print("[LLM] Summarizing logs using Google Gemma-2B-it...")
    summary = query_llm(prompt)
    print("[SUMMARY]\n", summary)
    return summary
    
# === TTS 변환 및 재생 ===
def text_to_speech(text, filename="summary.mp3"):
    try:
        tts = gTTS(text=text, lang="ko")
        tts.save(filename)
        os.system(f"mpg123 -q {filename}")
        print("[TTS] Summary spoken successfully.")
    except Exception as e:
        print(f"[TTS Error] {e}")


# === MQTT 콜백 ===
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[OK] Connected to broker")
        client.subscribe("project/#")
        client.subscribe("command/#")
        print("[SUB] Subscribed to project/# and command/#")
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

    # === 명령 트리거 ===
    if topic == "command/summary":
        print("[CMD] Summary request received → Generating report...")
        logs = fetch_logs(10)
        summary = summarize_logs(logs)
        print("[SUMMARY]\n", summary)
        text_to_speech(summary)

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

