import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish # STT 스레드에서 publish.single 사용
import pymysql
from datetime import datetime, timezone
from gtts import gTTS
import os
from openai import OpenAI
import sys
import re 
import json 
import threading # STT 기능을 별도 스레드에서 실행하기 위함
import speech_recognition as sr # STT 기능 추가
import time # sleep 함수 사용
import subprocess
from functools import wraps

# === Flask and SocketIO Imports (NEW) ===
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit, join_room

# === 전역 변수 및 초기화 ===
DB_CONN = None # 전역 DB 연결 객체
CURSOR = None # 전역 DB 커서 객체

# === DB 연결 (MariaDB) ===
DB_HOST = "localhost"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# 🚨🚨 MQTT 인증 정보 추가 (SERVER_USER 사용) 🚨🚨
MQTT_USERNAME = "SERVER_USER" # Mosquitto에 등록된 사용자 이름으로 변경
MQTT_PASSWORD = "sksk" # Mosquitto에 등록된 비밀번호로 변경

# === MQTT 설정 ===
BROKER = "0.0.0.0" # 브로커 IP 설정 필요 (Docker 환경 시 10.10.14.73 등)
PORT = 1883
TOPIC_BASE = "project/"   # 모듈 로그 접두사 (예: project/IMU/RAW)
COMMAND_TOPIC = "command/summary" # 항해일지 요약 명령
QUERY_TOPIC = "command/query" # 일반 질의 명령

# === Flask and SocketIO Initialization (NEW) ===
app = Flask(__name__)
# 웹 소켓 CORS 허용 (개발 환경을 위해 *로 설정)
app.config['SECRET_KEY'] = 'a_secure_secret_key_for_socketio'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- DB 관련 함수 ---
def connect_db():
    """전역 DB 연결 및 커서를 초기화합니다. DictCursor를 사용합니다."""
    global DB_CONN, CURSOR
    try:
        # DictCursor를 사용하여 DB 결과를 딕셔너리 형태로 반환하도록 설정 (웹페이지 처리에 용이)
        DB_CONN = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db=DB_NAME,
                                  cursorclass=pymysql.cursors.DictCursor) # DictCursor 사용
        CURSOR = DB_CONN.cursor()
        print(f"[DB INFO] MariaDB 연결 성공: {DB_NAME}")
    except pymysql.Error as e:
        print(f"[DB CRITICAL] MariaDB 연결 실패: {e}")
        sys.exit(1)

def close_db(client=None):
    """DB 연결을 종료합니다."""
    global DB_CONN, CURSOR
    if CURSOR:
        CURSOR.close()
        CURSOR = None
    if DB_CONN:
        DB_CONN.close()
        DB_CONN = None
    if client:
        print("[EXIT] Server stopped successfully.")

def insert_into_db(ts, module, object_type, risk_level, description, detail_json):
    """vision_data 테이블에 데이터를 삽입합니다."""
    # detail_json은 문자열로 변환
    detail_str = json.dumps(detail_json) if isinstance(detail_json, dict) else detail_json

    insert_query = """
        INSERT INTO vision_data (ts, module, object_type, risk_level, description, detail_json)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (ts, module, object_type, risk_level, description, detail_str)
    
    try:
        if CURSOR and DB_CONN:
            CURSOR.execute(insert_query, values)
            DB_CONN.commit()
        else:
            print("[DB WARN] DB 연결 또는 커서가 초기화되지 않았습니다. 로그 삽입 실패.")
    except Exception as e:
        print(f"[DB ERROR] 데이터 삽입 중 오류 발생: {e}")
        if DB_CONN:
             DB_CONN.rollback()

def get_initial_logs_from_db(limit=50):
    """[누락된 기능] DB에서 최신 로그를 limit 수만큼 가져와 딕셔너리 리스트로 반환합니다."""
    try:
        if CURSOR and DB_CONN:
            query = """
                SELECT ts, module, risk_level, description, detail_json
                FROM vision_data 
                ORDER BY ts DESC 
                LIMIT %s
            """
            CURSOR.execute(query, (limit,))
            logs = CURSOR.fetchall() 

            # datetime 객체를 ISO 8601 문자열로 변환하여 JSON 직렬화 문제를 해결합니다.
            for log in logs:
                if isinstance(log.get('ts'), datetime):
                    # 타임존 정보 포함하여 포맷팅
                    log['ts'] = log['ts'].astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                # detail_json이 문자열일 경우 JSON 객체로 파싱 시도 (웹페이지 처리를 위함)
                if log.get('detail_json') and isinstance(log['detail_json'], str):
                    try:
                        log['detail_json'] = json.loads(log['detail_json'])
                    except json.JSONDecodeError:
                        pass # 파싱 실패 시 문자열 유지

            return logs
        return []
    except Exception as e:
        print(f"[DB ERROR] 초기 로그 조회 실패: {e}")
        return []

# --- Flask Web Route (NEW) ---

@app.route('/')
def index():
    """메인 웹페이지를 렌더링합니다."""
    # index.html 파일이 'templates' 폴더에 있어야 합니다.
    return render_template('index.html')

@app.route('/api/logs/initial', methods=['GET'])
def get_initial_logs():
    """[추가됨] 웹페이지 로드 시 초기 로그 데이터를 제공합니다."""
    logs = get_initial_logs_from_db(limit=50) # 최근 50개 로그
    return jsonify(logs)

# === 오디오 디버깅 설정 ===
# STT 초기화 실패 시 어떤 장치가 사용 가능한지 확인하기 위한 변수
# 기본값은 None이며, STT가 실패하면 이 변수를 통해 사용 가능한 장치 정보를 출력합니다.
AUDIO_DEVICE_INFO = None 

# TTS 재생 중단 기능을 위한 전역 변수
TTS_PROCESS = None
TTS_LOCK = threading.Lock()

# === OpenAI 클라이언트 설정 ===
# 키는 환경 변수에서 자동 로드됩니다.
try:
    client_llm = OpenAI() 
except Exception as e:
    print(f"[LLM-SETUP] OpenAI 클라이언트 초기화 오류: {e}. API 키를 확인하세요.")
    client_llm = None # 초기화 실패 시 None으로 설정

# === 유틸리티 ===
def now_str():
    """UTC 시각을 'YYYY-MM-DD HH:MM:SS.ffffff' (마이크로초) 형식으로 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

# === DB 연결 함수 (연결이 끊어졌을 경우를 대비) ===
def get_db_connection():
    """DB 연결 객체를 생성하고 반환합니다. 연결 실패 시 None 반환."""
    try:
        # 전역 상수 DB_HOST, DB_USER 등을 사용합니다.
        db = pymysql.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, 
            database=DB_NAME, charset="utf8mb4"
        )
        return db
    except Exception as e:
        print(f"[{now_str()}] [DB-ERROR] 연결 실패: {e}")
        return None
    
# DB 연결 확인 및 재연결 함수
def ensure_db_connection():
    """DB 연결 확인 및 재연결 후 글로벌 CURSOR를 갱신합니다."""
    global DB_CONN, CURSOR
    try:
        # DB_CONN이 None일 경우에도 예외 처리
        if DB_CONN is None:
             raise pymysql.err.InterfaceError("DB_CONN is None")
        DB_CONN.ping(reconnect=True)
    except Exception as e:
        print(f"[{now_str()}] [DB-WARN] 기존 연결 ping 실패. 재연결 시도.")
        # ping 재연결마저 실패했거나 연결 객체 자체가 문제가 있을 경우, 
        # get_db_connection을 통해 완전히 새로운 연결을 시도
        new_conn = get_db_connection()
        if new_conn:
            DB_CONN = new_conn
        else:
            print(f"[{now_str()}] [DB-CRITICAL] DB 재연결 최종 실패.")
            raise

    # CURSOR를 반드시 갱신하거나 새로 생성
    try:
        if CURSOR and CURSOR.connection != DB_CONN:
             CURSOR.close()
        CURSOR = DB_CONN.cursor()
    except Exception:
        # CURSOR가 아직 초기화되지 않았거나 닫혔다면, 새로 생성
        CURSOR = DB_CONN.cursor()
    
# === 키=값; 형태의 문자열을 딕셔너리로 파싱 ===
def parse_payload_to_dict(payload: str) -> dict:
    """'키=값;키=값' 형태의 문자열을 딕셔너리로 파싱합니다. JSON 우선 파싱."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # JSON이 아니면 기존 키=값; 로직을 유지합니다. 
        data = {}
        if "|" in payload:
            payload = payload.split("|", 1)[-1].strip()
        pairs = payload.split(';')
        for pair in pairs:
            if '=' in pair:
                k, v = pair.split('=', 1)
                data[k.strip()] = v.strip()
        return data

def clean_tts_text(text: str) -> str:
    """
    TTS 재생을 위해 불필요한 마크다운 문자를 제거하되, 한글/구두점은 유지합니다.
    """
    cleaned_text = text.replace('**', '').replace('*', '').replace('#', '')
    # 한글, 영문, 숫자, 공백, 자주 쓰는 구두점만 남기고 모두 제거
    cleaned_text = re.sub(r'[^\w\s\.\,\!\?ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# === DB 연결 초기화 (함수 정의 후 실행되어야 함) ===
DB_CONN = get_db_connection()
if DB_CONN is None:
    print("[CRITICAL] DB 연결 실패. 서버를 종료합니다.")
    sys.exit(1)
CURSOR = DB_CONN.cursor()

# --- 오디오 장치 디버깅 및 확인 함수 ---
def list_audio_devices(recognizer: sr.Recognizer):
    """시스템이 인식하는 모든 마이크/오디오 장치 목록을 출력합니다."""
    global AUDIO_DEVICE_INFO
    
    print("\n--- 🎙️ 인식된 오디오 장치 목록 (PyAudio 기준) ---")
    
    try:
        # PyAudio가 초기화되면 recognizer.pyaudio_module을 통해 접근 가능
        p = recognizer.pyaudio_module 
        
        # PyAudio 모듈이 로드되지 않았을 경우 (예외가 발생한 경우)
        if p is None:
            print("❌ PyAudio 모듈을 로드할 수 없습니다. speech_recognition 라이브러리 설치를 확인하세요.")
            return

        info = p.PyAudio().get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        if numdevices == 0:
            print("⚠️ **오디오 장치가 전혀 감지되지 않았습니다.** ALSA 드라이버 상태를 확인하거나 snd-dummy 모듈을 로드해야 합니다.")
            return
            
        AUDIO_DEVICE_INFO = []
        
        # 목록 출력 및 정보 저장
        for i in range(0, numdevices):
            device_info = p.PyAudio().get_device_info_by_host_api_device_index(0, i)
            # 녹음 장치(마이크)만 필터링합니다.
            if (device_info.get('maxInputChannels')) > 0: 
                print(f"✅ Input Device index: {i} - {device_info.get('name')}")
                AUDIO_DEVICE_INFO.append(device_info)
                
        print("---------------------------------------------------------")
        if not AUDIO_DEVICE_INFO:
             print("⚠️ **녹음 가능한 입력 장치(마이크)가 없습니다.**")

    except Exception as e:
        print(f"❌ 오디오 장치 목록을 가져오는 중 오류 발생: {e}")

def check_microphone(r: sr.Recognizer):
    """마이크가 시스템에 연결되어 있고 소리를 감지하는지 확인합니다."""
    
    print("\n--- 🎙️ 마이크 테스트 시작 ---")
    
    # --------------------------------------------------------------------------------
    # TODO: [사용자 지정] 여기에 STT를 시도할 마이크 장치 인덱스를 넣어보세요.
    # 이전 단계에서 출력된 목록에서 'Dummy'나 실제 마이크 인덱스를 확인 후 여기에 입력합니다.
    # 예시: DEVICE_INDEX = 3
    # --------------------------------------------------------------------------------
    DEVICE_INDEX = None # 기본값: 시스템 기본 마이크 사용

    try:
        # 장치 인덱스를 명시적으로 지정하거나, None(기본값)을 사용합니다.
        with sr.Microphone(device_index=DEVICE_INDEX, sample_rate=16000) as source:
            print("1. 마이크 연결 확인: 성공 (마이크 장치 접근 가능)")
            print("2. 주변 소음 캘리브레이션 중 (1.0초)...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            print("3. 마이크 활성화 완료. 3초 동안 소리를 들어봅니다.")
            
            try:
                # 짧게 소리를 들어서 스트림이 유효한지 확인 (실제 녹음 시도)
                audio = r.listen(source, timeout=3, phrase_time_limit=3) 
                
                if audio and len(audio.frame_data) > 0:
                    print("✅ 마이크 테스트 성공: 소리 감지 및 입력 데이터 확보 완료.")
                    return True
                else:
                    print("⚠️ 마이크 테스트 경고: 마이크가 연결되었으나, 3초 동안 유효한 소리를 감지하지 못했습니다.")
                    return True # 연결은 되었으므로, 일단 True를 반환하여 루프를 시도합니다.
            
            except sr.WaitTimeoutError:
                print("⚠️ 마이크 테스트 경고: 마이크가 연결되었으나, 3초 동안 유효한 소리를 감지하지 못했습니다. (조용한 환경일 수 있음)")
                return True # 연결은 되었으므로, 일단 True를 반환하여 루프를 시도합니다.
                
    except Exception as e:
        # [Errno -9999] Unanticipated host error 등 치명적인 오류 발생 지점
        print(f"❌ 마이크 테스트 치명적 실패: 오디오 스트림을 열 수 없습니다 ({e})")
        # 실패 시 장치 목록을 출력하여 디버깅을 돕습니다.
        list_audio_devices(r)
        return False

def check_speaker():
    """gTTS를 통해 짧은 음성을 생성하고 mpv로 재생하여 스피커 연결을 확인합니다."""
    TEST_FILENAME = "test_audio_output.mp3"
    TEST_TEXT = "테스트를 위해 스피커 출력을 확인합니다."
    
    print("\n--- 🔊 스피커 테스트 시작 ---")
    
    try:
        tts = gTTS(text=TEST_TEXT, lang="ko")
        tts.save(TEST_FILENAME)
        print(f"1. TTS 파일 생성 완료: {TEST_FILENAME}")
        
        # mpv 명령어 실행 (PipeWire 오류와 독립적)
        print("2. 스피커로 테스트 음성 재생 중...")
        os.system(f"mpv --no-terminal --volume=100 {TEST_FILENAME}") 
        
        print("✅ 스피커 테스트 성공: 음성 출력을 확인했습니다. (TTS/TTS 기능 사용 가능)")
        return True

    except Exception as e:
        print(f"❌ 스피커 테스트 실패: 음성 파일 생성 또는 재생 오류. 'gTTS' 또는 'mpv' 설치를 확인하세요. ({e})")
        return False
    finally:
        if os.path.exists(TEST_FILENAME):
            os.remove(TEST_FILENAME)

# === DB 저장 함수 (DB_CONN, CURSOR 사용) ===
def save_event_log(module: str, action: str, full_payload: str):
    """events 테이블에 일반 로그, STT, 모든 CRITICAL/WARNING 로그를 저장"""
    try:
        ensure_db_connection()

        now = now_str()
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        CURSOR.execute(sql, (module, action, full_payload, now))
        DB_CONN.commit()
        
        # 🚨 SocketIO: 모든 이벤트 로그를 웹 대시보드에 전송 (NEW)
        log_data = {
            "ts": now,
            "module": module,
            "action": action,
            "payload": full_payload,
        }
        socketio.emit('event_log', log_data)
        
        print(f"[{now}] [DB-OK] Log saved to events: ({module}) {action}")
    except Exception as e:
        print(f"[{now}] [DB-ERROR] events 테이블 저장 실패: {e}")
# 'module' 인수를 사용하여 AD/PE/VISION을 명확히 구분
def save_vision_data(module: str, action: str, payload_dict: dict):
    """
    vision_data 테이블에 VISION/AD/PE 결과를 저장합니다.
    'detections' 리스트를 우선 사용하고, 없으면 'details'를 fallback으로 시도합니다.
    """
    try:
        ensure_db_connection()

        now = now_str()
        # payload 안의 detections 리스트 우선, 없으면 details로 대체
        detections = payload_dict.get('detections')
        if detections is None:
            detections = payload_dict.get('details', [])

        if not detections:
            print(f"[{now}] [WARN] No detections/details found in {module} payload (action={action}). Skipping DB insert.")
            return

        sql = """
            INSERT INTO vision_data
            (ts, module, object_type, risk_level, description, detail_json)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        records_inserted = 0
        for detection in detections:
            # 안전하게 키들을 추출 (여러 포맷 대비)
            object_type = detection.get('object_type') or detection.get('object') or detection.get('type') or 'UNKNOWN'
            # risk_level may be under various keys
            risk_level = int(detection.get('risk_level', detection.get('level', detection.get('risk', 0))) or 0)
            description = detection.get('description') or detection.get('action') or detection.get('posture') or detection.get('zone') or ''
            confidence = float(detection.get('confidence', detection.get('score', 0.0) or 0.0))
            track_id = detection.get('track_id') or detection.get('id')

            detail_json = json.dumps(detection, ensure_ascii=False)

            CURSOR.execute(sql, (
                now,
                module,
                object_type,
                risk_level,
                description,
                detail_json,
            ))
            records_inserted += 1

        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Saved {records_inserted} records to vision_data from {module} ({action}).")

    except Exception as e:
        try:
            DB_CONN.rollback()
        except pymysql.err.InterfaceError:
             pass 
        except Exception:
             pass
             
        print(f"[{now}] [DB-ERROR] events 테이블 저장 실패: {e}")

def save_imu_raw_data(payload_dict: dict):
    """imu_data 테이블에 연속적인 Pitch/Roll/Yaw 데이터를 저장"""
    try:
        ensure_db_connection()
        
        now = now_str()
        
        # 클라이언트가 보낸 roll, pitch, yaw 키를 사용합니다.
        roll = float(payload_dict.get('roll', 0.0) or payload_dict.get('roll_angle', 0.0)) 
        pitch = float(payload_dict.get('pitch', 0.0))
        yaw = float(payload_dict.get('yaw', 0.0))
        
        sql = "INSERT INTO imu_data (ts, pitch, roll, yaw) VALUES (%s, %s, %s, %s)"
        # 순서를 DB 테이블 순서에 따라 Pitch, Roll, Yaw 순으로 맞춥니다.
        CURSOR.execute(sql, (now, pitch, roll, yaw)) 
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Raw data saved to imu_data: R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f}")
    except Exception as e:
        print(f"[DB-ERROR] imu_data 테이블 저장 실패: {e}")

# === LLM/TTS 로직 함수 (DB_CONN, CURSOR 사용) ===

def query_llm(prompt: str) -> str:
    """OpenAI API를 사용하여 LLM에 질문하고 응답을 받습니다."""
    global client_llm
    if client_llm is None:
        return "⚠️ LLM 클라이언트가 초기화되지 않았습니다. API 키를 확인하세요."
        
    try:
        # LLM 시스템 프롬프트: 답변 시 마크다운 기호를 사용하지 않고 평문으로만 응답하도록 강제
        messages = [
             {"role": "system", "content": "너는 선박 항해 보조관이야. 로그를 분석하여 간결하고 명확하게 한국어로 브리핑해줘. 답변 시 마크다운 기호(\\#, \\*, \\- 등)를 절대 사용하지 말고, 문장 끝에 마침표를 제외한 쉼표나 기타 구두점의 사용을 최소화하며 평문으로만 응답해야 해."},
             {"role": "user", "content": prompt}
        ]
        response = client_llm.chat.completions.create(
             model="gpt-4o-mini",
             messages=messages,
             max_tokens=300,
             temperature=0.7,
        )
        result = response.choices[0].message.content
        print("[LLM OK] Response received.")
        return result
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return "⚠️ LLM 요청 중 오류 발생."

# === 로그 불러오기 및 IMU 통계 가져오기 ===
def fetch_logs(minutes=15):
    """DB에서 최근 minutes분 동안의 이벤트 로그와 IMU 통계를 가져옵니다."""
    logs = []
    imu_stats = {
        'max_roll': 0.0,
        'min_roll': 0.0,
        'latest_yaw': 0.0
    }
    
    try:
        ensure_db_connection() # 로그 조회 전 연결 재확인
        
        # 1. 이벤트 로그 가져오기 (events 테이블)
        sql_events = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        CURSOR.execute(sql_events, (minutes,)) 
        rows = CURSOR.fetchall()
        logs = [f"[{r[0]}] ({r[1]}) {r[2]} → {r[3]}" for r in rows]
        print(f"[DB] Retrieved {len(logs)} event logs.")

        # 2. IMU 통계 가져오기 (imu_data 테이블)
        # 최대/최소 Roll (기울기)
        sql_roll = """
            SELECT MAX(roll), MIN(roll)
            FROM imu_data
            WHERE ts >= NOW() - INTERVAL %s MINUTE
        """
        CURSOR.execute(sql_roll, (minutes,))
        max_roll, min_roll = CURSOR.fetchone()
        imu_stats['max_roll'] = max_roll if max_roll is not None else 0.0
        imu_stats['min_roll'] = min_roll if min_roll is not None else 0.0

        # 최신 Yaw (현재 방향)
        sql_yaw = """
            SELECT yaw
            FROM imu_data
            WHERE ts >= NOW() - INTERVAL %s MINUTE
            ORDER BY ts DESC 
            LIMIT 1
        """
        CURSOR.execute(sql_yaw, (minutes,))
        latest_yaw_result = CURSOR.fetchone()
        imu_stats['latest_yaw'] = latest_yaw_result[0] if latest_yaw_result else 0.0
        
        print("[DB] Retrieved IMU statistics.")
        
    except Exception as e:
        print(f"[DB-ERROR] Log or IMU data fetching failed: {e}")
        logs = [f"최근 {minutes}분 동안 로그 불러오기 실패."]
        
    return logs, imu_stats
    
# === LLM 요약 (응답 스타일 강제) ===
def summarize_logs(logs, imu_stats, minutes):
    """로그 목록과 IMU 통계를 LLM에 전달하여 요약 보고서를 생성합니다."""
    text = "\n".join(logs)
    
    # LLM에게 전달할 IMU 통계 정보
    imu_context = f"""
    [선박 통계 (최근 {minutes}분)]:
    - 최대 롤(기울기): {imu_stats['max_roll']:.2f}도
    - 최소 롤(기울기): {imu_stats['min_roll']:.2f}도
    - 현재 추정 방향 (Yaw): {imu_stats['latest_yaw']:.2f}도
    """
    
    # LLM 사용자 프롬프트: 4가지 규칙을 명시적으로 요구
    prompt = f"""
    다음은 선박 통계와 항해 이벤트 로그입니다:

    {imu_context}
    
    [항해 이벤트 로그]:
    {text}

    위 정보를 분석하여 한국어로 간결하고 명확하게 브리핑해주세요. 응답은 오직 하나의 문단 형태로 작성해야 하며, 다음 4가지 정보를 반드시 포함해야 합니다:
    1. 선박의 일반적인 상태 (위 IMU 통계를 활용하여 최대 기울기 및 현재 방향 포함).
    2. 최근 {minutes}분간 'ALERT' 등 발생한 주요 이벤트 또는 특이사항.
    3. 카메라나 레이더 모듈(VISION, AD, PE)을 통해 감지된 위험 상황 관련 요약.
    4. 발생한 문제에 대해 조치된 사항이나 필요한 추가 조치. (로그에 조치 내용이 없으면 '현재 조치된 사항은 없습니다.' 등으로 언급).

    답변 시 마크다운 기호(\\#, \\*, \\- 등)는 절대 사용하지 말고, 문장 끝에 마침표를 제외한 쉼표나 기타 구두점의 사용을 최소화하며 평문으로만 응답해야 합니다.
    """
    print("[LLM] Summarizing logs using GPT-4o mini...")
    summary = query_llm(prompt)
    print("[SUMMARY]\n", summary)
    return summary
    
# === TTS 변환 및 재생 ===
def text_to_speech(text, filename="summary.mp3"):
    """TTS 재생. 기존 재생 중이면 중단 후 새로 재생"""
    global TTS_PROCESS
    
    # 1. 프로세스 생성 (mpv)
    new_tts_process = None 
    
    with TTS_LOCK:
        if TTS_PROCESS and TTS_PROCESS.poll() is None:
            # 기존 TTS 중단
            TTS_PROCESS.terminate()
            TTS_PROCESS.wait()
            TTS_PROCESS = None
        try:
            clean_text = clean_tts_text(text)
            tts = gTTS(text=clean_text, lang="ko")
            tts.save(filename)
            
            # mpv 프로세스를 지역 변수에 저장
            new_tts_process = subprocess.Popen(["mpv", "--no-terminal", "--volume=100", "--speed=1.3", filename])
            TTS_PROCESS = new_tts_process # 전역 변수 업데이트
            
        except Exception as e:
            print(f"[TTS Error] {e}")
            
    # 2. 프로세스 외부에서 대기 (LOCK을 오래 잡지 않기 위해)
    # 🚨 생성된 프로세스만 대기하도록 수정
    if new_tts_process:
        new_tts_process.wait() # mpv 프로세스가 끝날 때까지 블록킹
        # 재생 완료 후 전역 변수 초기화 (선택적)
        with TTS_LOCK:
            if TTS_PROCESS == new_tts_process:
                TTS_PROCESS = None

# =======================================================================
# === [STT/음성 명령] 스레드 로직 ===
# =======================================================================

def parse_speech_command(text: str) -> tuple[str, str]:
    """
    음성 텍스트를 분석하여 명령 토픽과 페이로드를 결정합니다.
    """
    text_lower = text.lower()
    
    # 1. 요약/보고 명령 감지
    summary_keywords = ["요약해줘", "보고해줘", "브리핑해줘", "일지", "요약"]
    if any(keyword in text_lower for keyword in summary_keywords):
        
        # '최근 N분'에서 N 추출
        match = re.search(r'(\d+)\s*(분|시간)', text_lower)
        minutes = 15 # 기본값: 15분
        
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            
            if unit == "시간":
                minutes = value * 60
            else: # "분"
                minutes = value
        
        # 서버는 payload로 분(minutes) 값만 받습니다.
        return COMMAND_TOPIC, str(minutes)

    # 2. 일반 질문 명령
    else:
        # 일반 질문은 query 토픽으로 그대로 전송합니다.
        return QUERY_TOPIC, text

def stt_listening_loop():
    """마이크 입력을 받고 음성을 텍스트로 변환하여 MQTT로 전송하는 독립 루프입니다."""
    r = sr.Recognizer()
    
    # 🚨🚨🚨 수정 1: TTS 중단 로직은 루프 안으로 이동하거나, 'text'가 필요 없도록 수정해야 합니다. 
    # STT 스레드 시작 시 TTS 중단은 비논리적이므로, 이 부분은 제거하는 것이 맞습니다.
    # 단, TTS 재생 중 '그만 말하라'는 명령은 루프 안에서 처리해야 합니다.

    # MQTT publish는 독립 스레드에서 publish.single을 사용합니다.
    mqtt_broker = BROKER 
    
    # ----------------------------------------------------------------------
    # TODO: [사용자 지정] 여기에 STT를 시도할 마이크 장치 인덱스를 넣어보세요.
    # check_microphone 실행 후 출력된 목록에서 'Dummy' 장치 인덱스를 확인하세요.
    # ----------------------------------------------------------------------
    DEVICE_INDEX = None # 기본값: 시스템 기본 마이크 사용

    # 마이크 설정 및 캘리브레이션 (STT 성공을 위한 try-except 블록)
    try:
        # 장치 인덱스를 명시적으로 지정하거나, None(기본값)을 사용합니다.
        with sr.Microphone(device_index=DEVICE_INDEX, sample_rate=16000) as source:
            print("[STT-THREAD] Ambient noise calibrating...")
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("[STT-THREAD] Setup complete. Starting speech recognition loop...")
    
    except Exception as e:
        # 초기화 중 치명적인 오류 발생 (예: Errno -9999)
        # ... (이전 코드와 동일한 오류 처리 로직 유지)
        print(f"[CRITICAL] STT Initialization Error (Microphone): {e}")
        return 

    while True:
        try:
            # 장치 인덱스를 루프 내부에서도 명시적으로 지정하여 사용합니다.
            with sr.Microphone(device_index=DEVICE_INDEX, sample_rate=16000) as source:
                print("\n[STT-THREAD] Listening for command (Say '최근 N분 요약해줘')...")
                # 음성 인식 대기 (최대 10초 발화 길이 제한)
                audio = r.listen(source, timeout=None, phrase_time_limit=10) 
            
            print("[STT-THREAD] Recognizing speech...")
            # 구글 STT를 사용하여 한국어(ko-KR)로 인식
            text = r.recognize_google(audio, language="ko-KR") 
            print("[STT-THREAD] You said:", text)

            # 🛑 '그만 말하라' 명령에 대한 TTS 중단 로직
            stop_keywords = ["그만", "멈춰", "중단", "정지", "닥쳐"]
            if any(keyword in text for keyword in stop_keywords):
                 with TTS_LOCK:
                    if TTS_PROCESS and TTS_PROCESS.poll() is None:
                        TTS_PROCESS.terminate()
                        TTS_PROCESS.wait()
                        print("[STT-THREAD] 🛑 TTS playback terminated by voice command.")
                        # 웹에도 중단 알림
                        socketio.emit('tts_status', {"status": "stopped", "message": "음성 명령으로 TTS 중단됨"})
                        continue 

            topic, payload = parse_speech_command(text)
            
            # MQTT 전송
            try:
                publish.single(topic, payload=payload, hostname=mqtt_broker, qos=1)
                print(f"[STT-THREAD] MQTT Published: {topic} -> {payload}")
                save_event_log("STT", "COMMAND", text)
            except Exception as e:
                print(f"[STT-THREAD] MQTT publish error: {e}")

        except sr.UnknownValueError:
            print("[STT-THREAD] Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"[STT-THREAD] Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"[STT-THREAD] An unexpected error occurred in STT loop: {e}")
            time.sleep(1) # 짧게 대기 후 재시도

# =======================================================================
# === [데이터 라우터] 핵심 로직 (SocketIO Emit 추가) ===
# =======================================================================

def process_and_save_data(msg):
    """
    수신된 MQTT 메시지를 분석하여 알맞은 테이블에 저장하고,
    필요 시 이벤트를 생성한 후 SocketIO로 웹 클라이언트에 데이터를 전송합니다.
    """

    # 1. 토픽 파싱
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    payload_dict = parse_payload_to_dict(payload)

    parts = topic.split('/')

    # 기본값: 안전하게 처리
    module = None
    action = None

    # common patterns:
    # - project/imu/RAW                -> parts = [project, imu, RAW]   (module=imu, action=RAW)
    # - project/vision/AD/RAW          -> parts = [project, vision, AD, RAW] (module=AD, action=RAW)
    # - project/vision/PE/ALERT        -> parts = [project, vision, PE, ALERT]

    if len(parts) >= 4 and parts[1].lower() == 'vision':
        # vision has an extra level: project/vision/<MODULE>/<ACTION>
        module = parts[2].upper()
        action = parts[3].upper()
    elif len(parts) >= 3:
        # regular 3-level topics: project/<MODULE>/<ACTION>
        module = parts[1].upper()
        action = parts[2].upper()
    else:
        # too short: ignore unless it's a command topic handled elsewhere
        if not topic.startswith("command/"):
            print(f"[{now_str()}] [WARN] Skipping short or unknown topic: {topic}")
        return

    # =======================================================
    # 2. 데이터 라우팅 및 저장 (ALERT 우선 처리)
    # =======================================================

    # 2-1. 🚨 ALERT 토픽 처리 (CRITICAL/WARNING 레벨)
    if action in ["ALERT", "CRITICAL"]:
        now = now_str()

        # 1️⃣ 기존 TTS 재생 중단
        global TTS_PROCESS, TTS_LOCK
        with TTS_LOCK:
            if 'TTS_PROCESS' in globals() and TTS_PROCESS and TTS_PROCESS.poll() is None:
                TTS_PROCESS.terminate()
                TTS_PROCESS.wait()
                print(f"[{now}] [TTS] 기존 TTS 재생 중단 완료")

        # 2️⃣ DB 저장
        save_event_log(module, action, payload)
        if module in ["VISION", "AD", "PE"]:
            save_vision_data(module, action, payload_dict)
            print(f"[{now}] [DB] ALERT/CRITICAL log saved to events AND vision_data: {module}/{action}")
        else:
            print(f"[{now}] [DB] ALERT/CRITICAL log saved to events: {module}/{action}")

        # 3️⃣ 긴급 알람 TTS 재생
        print(f"[{now}] [TTS] 긴급 알람 발화: {module} {action}")
    
        # 현재 시각을 형식에 맞게 준비 (HH:MM:SS)
        current_time_short = datetime.now(timezone.utc).strftime("%H:%M:%S")
        tts_text = f"긴급 알람 발생: {module} {action}" # 기본 폴백 텍스트

        if module == "IMU" and action == "ALERT":
            # IMU 센서 위험 각도 초과 알람 처리
            detected_angle = payload_dict.get('roll_angle', payload_dict.get('roll', 0.0))
            if detected_angle is not None:
                # '긴급 상황 발생! HH:MM:SS 선체 각도 15.5도. 위험 각도 초과. HH:MM:SS 선체 각도 15.5도. 위험 각도 초과'
                tts_text = f"긴급 상황 발생! {current_time_short} 선체 각도 {detected_angle}도. 위험 각도 초과. {current_time_short} 선체 각도 {detected_angle}도. 위험 각도 초과"

        elif module == "AD" and action == "ALERT":
            # 어노말리 디텍션 객체 위험 접근 감지 알람 처리
            # 'detections' 리스트의 첫 번째 객체 'object_type' 또는 'object'를 사용
            detections = payload_dict.get('detections', payload_dict.get('details', []))
            object_type = '미확인 객체'
            if detections and isinstance(detections, list):
                # 첫 번째 객체 정보 사용
                object_type = detections[0].get('object_type', detections[0].get('object', '미확인 객체'))

            # '긴급 상황 발생! HH:MM:SS 대형 컨테이너선 접근 중. HH:MM:SS 대형 컨테이너선 접근 중'
            tts_text = f"긴급 상황 발생! {current_time_short} {object_type} 접근 중. {current_time_short} {object_type} 접근 중"

        elif module == "PE" and action == "CRITICAL" and payload_dict.get('action') in ["fall", "down"]:
            # 갑판 낙상 사고 감지 알람 처리 (PE 모듈, action이 fall 또는 down인 경우)
            # '긴급 상황 발생! HH:MM:SS 갑판에서 낙상 사고 발생. HH:MM:SS 갑판에서 낙상 사고 발생'
            tts_text = f"긴급 상황 발생! {current_time_short} 갑판에서 낙상 사고 발생. {current_time_short} 갑판에서 낙상 사고 발생"

        # 4️⃣ TTS 재생
        text_to_speech(tts_text)

        # 4️⃣ SocketIO로 실시간 경고 전송 (NEW)
        alert_data = {
            "ts": now,
            "topic": topic,
            "payload": payload_dict
        }
        socketio.emit('alert_event', alert_data)
        
        print(f"[{now}] [DB/WEB] ALERT/CRITICAL processed: {module}/{action}")

    # 2-2. 🟢 RAW 토픽 처리 (INFO 레벨 - 연속 데이터)
    elif action == "RAW":
        now = now_str()
        if module == "IMU":
            save_imu_raw_data(payload_dict)
            # SocketIO로 IMU 데이터 실시간 전송 (NEW)
            socketio.emit('imu_update', {"ts": now, "data": payload_dict})
            print(f"[{now}] [DB/WEB] Saved and Emitted IMU RAW data.")

        # VISION 시스템의 모든 세부 모듈(AD, PE 포함) 데이터를 vision_data에 통합 저장합니다.
        elif module in ["VISION", "AD", "PE"]:
            save_vision_data(module, action, payload_dict)
            # SocketIO로 Vision/AD/PE RAW 데이터 전송 (NEW)
            socketio.emit('vision_raw_update', {"ts": now, "module": module, "data": payload_dict})
            print(f"[{now}] [DB/WEB] Saved and Emitted {module} RAW data.")

        else:
            print(f"[{now_str()}] [WARN] Unknown RAW module: {module}. Data discarded.")
        return

    # 2-3. 기타 일반 시스템/STT 이벤트 (events 테이블)
    else:
        # save_event_log 함수 내부에서 SocketIO로 전송됨
        save_event_log(module, action, payload)
        print(f"[{now_str()}] [LOG] Saved general log to events table. Module: {module}")

def on_connect(client, userdata, flags, rc):
    global DB_CONN, CURSOR
    print(f"[INFO] Connected with result code {rc}")
    if rc == 0:
        # 구독: project/ 하위의 모든 토픽
        client.subscribe(TOPIC_BASE + "#") 
        print(f"[INFO] Subscribed to topic: {TOPIC_BASE}#")
        # TTS 명령 토픽 구독
        client.subscribe(COMMAND_TOPIC) 
        client.subscribe(QUERY_TOPIC)
    else:
        print(f"[CRITICAL] Connection failed (rc: {rc}).")


# === [MQTT 콜백] 명령어 처리 후 데이터 라우팅을 'process_and_save_data'로 위임하는 진입점. ===
def on_message(client, userdata, msg):
    """MQTT 메시지 수신 시 호출되며, 모든 처리를 process_and_save_data 함수로 위임합니다."""
    try:
        # 모든 데이터 라우팅, DB 저장, SocketIO 전송, TTS 로직을 이 함수로 위임
        process_and_save_data(msg)
    except Exception as e:
        # process_and_save_data 내부에서 처리 못한 예외만 여기서 잡도록 수정
        print(f"[{now_str()}] [CRITICAL] Error in on_message/router: {e}")

    # 2. === 데이터 처리 로직을 새로운 함수로 위임 ===
    process_and_save_data(msg)
    

# === MQTT 클라이언트 및 메인 루프 ===
# MQTTv311 프로토콜 명시로 DeprecationWarning 해결
client = mqtt.Client(client_id="MarineServer", protocol=mqtt.MQTTv311) 

# MQTT 인증 정보 설정
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

client.on_connect = on_connect
client.on_message = on_message

# === 서버 시작 시 호출되는 메인 블록 ===
if __name__ == '__main__':
    
    # === 브로커 연결 ===
    print("[INFO] Connecting to broker...")
    try:
        client.connect(BROKER, PORT, 60)
    except Exception as e:
        print(f"[CRITICAL] MQTT 연결 실패: {e}")
        sys.exit(1)

    # 1. STT 리스닝 스레드 시작
    stt_recognizer = sr.Recognizer()
    microphone_test_result = check_microphone(stt_recognizer)
    
    if microphone_test_result:
        stt_thread = threading.Thread(target=stt_listening_loop)
        stt_thread.daemon = True # 메인 스레드 종료 시 함께 종료
        stt_thread.start()
    else:
        print("\n[WARN] 마이크 초기화 실패로 STT/TTS 기능 스레드는 시작되지 않았습니다.")

    # 2. 스피커 테스트 (TTS 기능 확인)
    check_speaker()
    
    # 3. MQTT 클라이언트 루프 시작 (Flask/SocketIO와 동시 실행)
    client.loop_start() 
    
    # 4. Flask/SocketIO 서버 실행 (메인 스레드 점유)
    print("[INFO] Starting Flask SocketIO server on http://0.0.0.0:5000")
    try:
        # debug=False로 설정하여 두 번 실행되는 것을 방지합니다 (Thread 때문에 중요)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False) 
    except KeyboardInterrupt:
        # Ctrl+C가 눌렸을 때 깔끔하게 종료
        print("\n[EXIT] Server is stopping gracefully...")
    except Exception as e:
        print(f"\n[CRITICAL] Flask/SocketIO 서버 실행 중 오류: {e}")

    finally:
        print("[EXIT] Server cleanup...")
        client.disconnect()
        if CURSOR:
            CURSOR.close() 
        if DB_CONN:
            DB_CONN.close()
        print("[EXIT] Server stopped successfully.")
