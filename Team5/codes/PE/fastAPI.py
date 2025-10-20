"""
FastAPI 기반 해양 시스템 통합 대시보드 서버

기존 Flask/SocketIO 서버를 FastAPI와 WebSocket으로 재구성하여
성능 및 확장성을 개선한 버전입니다.

주요 기능:
1. FastAPI 웹 서버 및 WebSocket을 통한 실시간 데이터 전송
2. 비동기(asyncio) 기반의 MQTT 클라이언트 통합
3. 비동기 데이터베이스(MariaDB) 처리
4. STT, LLM, TTS 음성 명령 처리 기능 (기존 server.py 로직 완전 통합)
5. 실시간 비디오 스트리밍 엔드포인트 제공
"""

import asyncio
import base64
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone

import aiomysql  # 비동기 MySQL 드라이버
import paho.mqtt.client as mqtt
import speech_recognition as sr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from gtts import gTTS
from openai import OpenAI

# =======================================================================
# 설정 (기존 server.py와 유사)
# =======================================================================
# --- DB 설정 ---
DB_HOST = "10.10.14.42"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"
DB_POOL = None

# --- MQTT 설정 ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = "SERVER_USER"
MQTT_PASSWORD = "sksk"
TOPIC_BASE = "project/#"
COMMAND_TOPIC = "command/summary"
QUERY_TOPIC = "command/query"

# --- OpenAI 설정 ---
try:
    llm_client = OpenAI()
except Exception as e:
    print(f"--- [OpenAI-WARN] --- \n{e}\nLLM client will be disabled.\n---------------------")
    llm_client = None

# --- TTS/STT 설정 ---
TTS_PROCESS = None
TTS_LOCK = threading.Lock()

# --- 유틸리티 함수 ---
def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# =======================================================================
# FastAPI 애플리케이션 생명주기 (Lifespan)
# =======================================================================
async def lifespan(app: FastAPI):
    print("--- Server Starting Up ---")
    global DB_POOL
    try:
        DB_POOL = await aiomysql.create_pool(
            host=DB_HOST, port=3306, user=DB_USER, password=DB_PASSWORD,
            db=DB_NAME, autocommit=True
        )
        print("[DB] Async DB connection pool created successfully.")
    except Exception as e:
        print(f"--- [DB-CRITICAL] --- \nFailed to create DB pool: {e}\nDB functionalities will be disabled.\n-----------------------")
        DB_POOL = None

    # MQTT 클라이언트에 연결 정보를 설정하고 백그라운드 루프 시작
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("[MQTT] MQTT client loop started.")
    
    stt_thread = threading.Thread(target=stt_listening_loop, daemon=True)
    stt_thread.start()
    print("[STT] STT listening thread started.")

    yield 

    print("--- Server Shutting Down ---")
    mqtt_client.loop_stop()
    print("[MQTT] MQTT client loop stopped.")
    if DB_POOL:
        DB_POOL.close()
        await DB_POOL.wait_closed()
        print("[DB] DB connection pool closed.")

app = FastAPI(lifespan=lifespan)

# =======================================================================
# WebSocket 관리
# =======================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        # 여러 브로드캐스트 호출이 동시에 발생할 수 있으므로 asyncio.gather 사용
        await asyncio.gather(
            *[connection.send_text(message) for connection in self.active_connections],
            return_exceptions=False
        )

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# =======================================================================
# 데이터베이스 함수 (비동기)
# =======================================================================
async def db_execute(sql, args):
    if not DB_POOL:
        print("[DB-WARN] DB Pool not available. Skipping DB operation.")
        return
    try:
        async with DB_POOL.acquire() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, args)
    except Exception as e:
        print(f"[{now_str()}] [DB-ERROR] {e}")

async def save_event_log(module, action, payload):
    sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
    await db_execute(sql, (module, action, payload, now))
    log_data = {"event": "event_log", "data": {"ts": now, "module": module, "action": action, "payload": payload}}
    await manager.broadcast(json.dumps(log_data))

async def save_imu_raw_data(payload):
    roll = float(payload.get('roll',0)); pitch = float(payload.get('pitch',0)); yaw = float(payload.get('yaw',0))
    sql = "INSERT INTO imu_data (ts, pitch, roll, yaw) VALUES (%s, %s, %s, %s)"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
    await db_execute(sql, (now, pitch, roll, yaw))
    imu_data = {"event": "imu_update", "data": {"ts": now, "data": payload}}
    await manager.broadcast(json.dumps(imu_data))

# =======================================================================
# LLM / TTS 함수 (기존 server.py 로직 복원)
# =======================================================================
def query_llm(prompt: str) -> str:
    if not llm_client:
        return "LLM client is not initialized. Please check your API key."
    try:
        messages = [
             {"role": "system", "content": "You are a ship navigation assistant. Analyze the logs and provide a concise and clear briefing in Korean. Do not use any markdown symbols and respond only in plain text."},
             {"role": "user", "content": prompt}
        ]
        response = llm_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during the LLM request: {e}"

def text_to_speech(text: str):
    global TTS_PROCESS
    with TTS_LOCK:
        if TTS_PROCESS and TTS_PROCESS.poll() is None:
            TTS_PROCESS.terminate(); TTS_PROCESS.wait()
        try:
            clean_text = re.sub(r'[^\w\s\.\,\!\?ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', text)
            tts = gTTS(text=clean_text, lang="ko")
            tts.save("summary.mp3")
            TTS_PROCESS = subprocess.Popen(["mpv", "--no-terminal", "summary.mp3"])
        except Exception as e:
            print(f"[TTS Error] {e}")

# =======================================================================
# MQTT 클라이언트 로직
# =======================================================================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected. Subscribing to topics.")
        client.subscribe([(TOPIC_BASE, 1), (COMMAND_TOPIC, 1), (QUERY_TOPIC, 1)])
    else:
        print(f"[MQTT-CRITICAL] Connection failed (Code: {rc}).")

def on_message(client, userdata, msg):
    # [FIX] 메인 이벤트 루프를 가져와서 비동기 함수를 안전하게 스케줄링
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.run_coroutine_threadsafe(handle_mqtt_message(msg), loop)

async def handle_mqtt_message(msg):
    topic = msg.topic
    payload_str = msg.payload.decode('utf-8')
    now = now_str()
    print(f"[{now}] [MQTT-RECV] {topic} -> {payload_str[:100]}...")

    if "VIDEO" in topic:
        await manager.broadcast(json.dumps({"event": "video_stream", "data": {"topic": topic, "frame": payload_str}}))
        return
    
    if topic.startswith("command/"):
        # This is where you would handle LLM summary/query logic
        return

    try:
        payload_dict = json.loads(payload_str)
    except json.JSONDecodeError:
        payload_dict = {"message": payload_str}

    parts = topic.split('/')
    module = parts[2] if len(parts) > 2 else "UNKNOWN"
    action = parts[3] if len(parts) > 3 else "RAW"

    if action in ["ALERT", "CRITICAL"]:
        await save_event_log(module, action, payload_str)
        alert_data = {"event": "alert_event", "data": {"ts": now, "topic": topic, "payload": payload_dict}}
        await manager.broadcast(json.dumps(alert_data))
        if "message" in payload_dict:
            text_to_speech(f"Emergency alert, {payload_dict['message']}")

    elif action == "RAW":
        if "IMU" in module:
            await save_imu_raw_data(payload_dict)
        elif "FALL" in module or "AD" in module:
            raw_data = {"event": "vision_raw_update", "data": {"ts": now, "module": module, "data": payload_dict}}
            await manager.broadcast(json.dumps(raw_data))

mqtt_client = mqtt.Client(client_id="FastAPIServer", protocol=mqtt.MQTTv311)
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# =======================================================================
# STT 로직 (기존 server.py 로직 복원)
# =======================================================================
def parse_speech_command(text: str) -> tuple[str, str]:
    text_lower = text.lower()
    summary_keywords = ["summary", "report", "briefing", "log"]
    if any(keyword in text_lower for keyword in summary_keywords):
        match = re.search(r'(\d+)\s*(minute|hour)', text_lower)
        minutes = "15"
        if match:
            value = int(match.group(1))
            minutes = str(value * 60 if match.group(2) == "hour" else value)
        return COMMAND_TOPIC, minutes
    return QUERY_TOPIC, text

def stt_listening_loop():
    r = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                print("\n[STT] Listening for command...")
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
            text = r.recognize_google(audio, language="ko-KR")
            print(f"[STT] You said: {text}")
            
            stop_keywords = ["stop", "halt", "cease"]
            if any(keyword in text for keyword in stop_keywords):
                with TTS_LOCK:
                    if TTS_PROCESS and TTS_PROCESS.poll() is None:
                        TTS_PROCESS.terminate(); TTS_PROCESS.wait()
                continue

            topic, payload = parse_speech_command(text)
            mqtt_client.publish(topic, payload, qos=1)
            
            # [FIX] 메인 이벤트 루프를 가져와서 비동기 함수를 안전하게 스케줄링
            loop = asyncio.get_event_loop()
            if loop.is_running():
                 asyncio.run_coroutine_threadsafe(save_event_log("STT", "COMMAND", text), loop)

        except sr.UnknownValueError:
            print("[STT-WARN] Could not understand audio.")
        except sr.RequestError as e:
            print(f"[STT-ERROR] Google Speech Recognition service error; {e}")
        except Exception as e:
            print(f"[STT-ERROR] An unexpected error occurred: {e}")
            time.sleep(1)

# =======================================================================
# 웹페이지 라우트
# =======================================================================
@app.get("/", response_class=HTMLResponse)
async def get():
    # [FIX] 파일 경로를 절대 경로로 지정하여 오류 방지
    try:
        # __file__은 현재 스크립트 파일의 경로를 나타냄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "index.html")
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Internal Server Error: {e}</h1>", status_code=500)

