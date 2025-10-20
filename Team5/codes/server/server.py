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

# === DB 연결 (MariaDB) ===
DB_HOST = "localhost"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# 🚨🚨 SERVER_USER 인증 정보 추가 🚨🚨
MQTT_USERNAME = "SERVER_USER"      # 등록된 SERVER 사용자 이름
MQTT_PASSWORD = "sksk"  # 등록된 SERVER 사용자 비밀번호

# === MQTT 설정 ===
BROKER = "10.10.14.73"
PORT = 1883
TOPIC_BASE = "project/"   # 모듈 로그 접두사 (예: project/IMU/RAW)
COMMAND_TOPIC = "command/summary" # 항해일지 요약 명령
QUERY_TOPIC = "command/query" # 일반 질의 명령
# GUI 실시간 로그 전송용 토픽
GUI_TOPIC_LOG = "project/log/RAW"
LOGBOOK_TOPIC = "project/log/LOGBOOK"
STATUS_TOPIC = "project/status"

# === 오디오 디버깅 설정 ===
# STT 초기화 실패 시 어떤 장치가 사용 가능한지 확인하기 위한 변수
# 기본값은 None이며, STT가 실패하면 이 변수를 통해 사용 가능한 장치 정보를 출력합니다.
AUDIO_DEVICE_INFO = None 

# TTS 재생 중단 기능을 위한 전역 변수
TTS_PROCESS = None
TTS_LOCK = threading.Lock()

# === OpenAI 클라이언트 설정 ===
client_llm = OpenAI() # 키는 환경 변수에서 자동 로드됩니다.

# === 유틸리티 ===
def now_str():
    """UTC 시각을 'YYYY-MM-DD HH:MM:SS'"""
    # 초 단위가 아닌 마이크로초 단위까지 포함하여 고유성을 높입니다. (Duplicate Entry 방지)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

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
    
    # 1. 연결 객체 유효성 확인 및 재연결 시도
    if DB_CONN is None:
        new_conn = get_db_connection()
        if new_conn:
            DB_CONN = new_conn
        else:
            print(f"[{now_str()}] [DB-CRITICAL] DB 재연결 최종 실패.")
            raise ConnectionError("DB connection could not be established.")

    try:
        # 연결이 끊어졌는지 확인하고, 끊겼다면 자동 재연결 시도
        DB_CONN.ping(reconnect=True)
    except Exception as e:
        print(f"[{now_str()}] [DB-WARN] 기존 연결 ping 실패. 재연결 시도: {e}")
        # ping 재연결마저 실패했거나 연결 객체 자체가 문제가 있을 경우, 
        # get_db_connection을 통해 완전히 새로운 연결을 시도
        new_conn = get_db_connection()
        if new_conn:
            DB_CONN = new_conn
        else:
            print(f"[{now_str()}] [DB-CRITICAL] DB 재연결 최종 실패.")
            raise ConnectionError("DB connection ping failed and could not reconnect.")
    
    # 2. 🚨 CURSOR 갱신/재생성 (매우 중요: 특히 재연결 시)
    try:
        # 기존 CURSOR가 닫혔거나 연결이 변경되었을 경우 새로 생성
        if CURSOR is None or CURSOR.connection != DB_CONN:
             if CURSOR:
                 CURSOR.close()
             CURSOR = DB_CONN.cursor()
    except Exception:
        # 어떤 이유로든 CURSOR 처리에 문제 발생 시 새로 생성
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
        print("   (이 오류는 PyAudio가 초기화 실패 시 흔히 발생합니다.)")


def check_microphone(r: sr.Recognizer):
    """마이크가 시스템에 연결되어 있고 소리를 감지하는지 확인합니다."""
    
    print("\n--- 🎙️ 마이크 테스트 시작 ---")
    
    # --------------------------------------------------------------------------------
    # TODO: [사용자 지정] 여기에 STT를 시도할 마이크 장치 인덱스를 넣어보세요.
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

def get_compass_direction(yaw_angle: float) -> str:
    """Yaw 각도를 8방향 나침반 문자로 변환합니다 (N, NE, E, SE, S, SW, W, NW)."""
    # 0도 ~ 360도 범위로 보정 (IMU 데이터는 보통 이 범위에 있으나, 안전을 위해)
    yaw_angle = yaw_angle % 360
    
    # 22.5도 간격으로 8방향 구분
    directions = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    # 22.5를 더한 후 45로 나누어 인덱스를 얻습니다. (북쪽(0) 주변을 처리하기 위함)
    index = int((yaw_angle + 22.5) // 45) % 8
    
    return directions[index]

def publish_logbook_entries(mqtt_client):
    try:
        conn = pymysql.connect(
            host="localhost",
            user="marine_user",
            password="sksk",  # 실제 비밀번호
            database="marine_system",
            charset="utf8mb4"
        )
        cur = conn.cursor()
        cur.execute("SELECT * FROM logbook ORDER BY id DESC LIMIT 10;")
        rows = cur.fetchall()
        conn.close()

        entries = []
        for r in rows:
            entries.append({
                "id": r[0],
                "log_dt": str(r[1]),
                "sail_time": str(r[2]),
                "wind_dir": r[3],
                "wind_spd": r[4],
                "weather": r[5],
                "on_route": bool(r[6]),
                "on_notes": r[7],
                "ex_notes": r[8]
            })

        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module": "SERVER",
            "type": "LOGBOOK",
            "entries": entries
        }

        mqtt_client.publish(LOGBOOK_TOPIC, json.dumps(payload, ensure_ascii=False))
        print(f"[{datetime.now()}] ✅ 항해일지(LOGBOOK) 데이터 발행 완료 ({len(entries)}건)")

    except Exception as e:
        print(f"[LOGBOOK ERROR] {e}")

def save_llm_report(report_text: str):
    """LLM이 생성한 보고서를 logbook 테이블의 on_notes 필드에 저장합니다."""
    try:
        ensure_db_connection()
        
        now = now_str()
        
        # logbook 테이블에 삽입하는 SQL 쿼리 (보고서 텍스트만 on_notes에 저장)
        sql = """
            INSERT INTO logbook 
            (sail_time, wind_dir, wind_spd, weather, on_route, on_notes, ex_notes) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # NOTE: 보고서 전용 항목이므로 나머지 필드는 NULL로 채웁니다.
        CURSOR.execute(sql, (
            None,      # sail_time
            'LLM_REPORT', # wind_dir (보고서 식별자로 활용)
            None,      # wind_spd
            'OK',      # weather (기록이 성공했음을 표시)
            1,         # on_route (기본값)
            report_text, # <<<< LLM 요약 보고서 저장!
            None       # ex_notes
        ))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] ✅ LLM Report saved to logbook.")

        # GUI에 최신 logbook 데이터를 발행하여 갱신
        publish_logbook_entries(client)

    except Exception as e:
        try:
            DB_CONN.rollback()
        except Exception:
             pass
        print(f"[{now}] [DB-ERROR] ❌ logbook 테이블 저장 실패 (LLM Report): {e}")

# === DB 저장 함수 (DB_CONN, CURSOR 사용) ===
def save_event_log(module: str, action: str, full_payload: str):
    """events 테이블에 일반 로그, STT, 모든 CRITICAL/WARNING 로그를 저장"""
    try:
        global client
        ensure_db_connection()

        now = now_str()
        sql = "INSERT INTO events (module, action, payload, ts) VALUES (%s, %s, %s, %s)"
        CURSOR.execute(sql, (module, action, full_payload, now))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Log saved to events: ({module}) {action}")

        # GUI 실시간 전송 추가
        gui_payload = {
            "ts": now,
            "module": module,
            "action": action,
            "payload": full_payload
        }
        client.publish(GUI_TOPIC_LOG, json.dumps(gui_payload, ensure_ascii=False))

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
        gui_summary = []
        for detection in detections:
            # 안전하게 키들을 추출 (여러 포맷 대비)
            object_type = detection.get('object_type') or detection.get('object') or detection.get('type') or 'UNKNOWN'
            # risk_level may be under various keys
            risk_level = int(detection.get('risk_level', detection.get('level', detection.get('risk', 0))) or 0)
            description = detection.get('description') or detection.get('action') or detection.get('posture') or detection.get('zone') or ''

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

            gui_summary.append({
                "object": object_type,
                "risk": risk_level,
                "desc": description
            })

        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Saved {records_inserted} records to vision_data from {module} ({action}).")

        # GUI 실시간 전송
        gui_payload = {
            "ts": now,
            "module": module,
            "action": action,
            "detections": gui_summary
        }
        client.publish(GUI_TOPIC_LOG, json.dumps(gui_payload, ensure_ascii=False))

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
        global client
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
        
        # Roll 해석: 좌현(Port: -) 또는 우현(Starboard: +)
        roll_desc = f"{abs(roll):.2f}° " + ("(우현 기울임)" if roll >= 0 else "(좌현 기울임)")
        
        # Pitch 해석: 선수 들림(Up: +) 또는 선수 숙임(Down: -)
        pitch_desc = f"{abs(pitch):.2f}° " + ("(선수 들림)" if pitch >= 0 else "(선수 숙임)")
        
        # Yaw 해석: 방위각을 나침반 방향으로 변환 (예: 45° -> 북동)
        # ⚠️ (여기서는 간단히 각도만 표시하고, GUI에서 더 복잡한 변환을 수행할 수 있습니다.)
        direction = get_compass_direction(yaw)
        yaw_desc = f"{yaw:.2f}° ({direction})"

        # GUI 실시간 전송
        gui_payload = {
            "ts": now,
            "module": "IMU",
            "action": "RAW",
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "roll_desc": roll_desc,
            "pitch_desc": pitch_desc,
            "yaw_desc": yaw_desc
        }
        client.publish(GUI_TOPIC_LOG, json.dumps(gui_payload, ensure_ascii=False))

    except Exception as e:
        print(f"[DB-ERROR] imu_data 테이블 저장 실패: {e}")

def save_frame_data(module, base64_str):
    """카메라 프레임(Base64 인코딩된 이미지) 저장"""
    try:
        ensure_db_connection()
        now = now_str()
        
        sql = "INSERT INTO frames (ts, module, frame_base64) VALUES (%s, %s, %s)"
        CURSOR.execute(sql, (now, module, base64_str))
        DB_CONN.commit()
        print(f"[{now}] [DB-OK] Frame saved to frames: ({module})")

    except Exception as e:
        print(f"[DB-ERROR] ❌ Failed to save frame ({module}): {e}")

# === LLM/TTS 로직 함수 (DB_CONN, CURSOR 사용) ===

def query_llm(prompt: str) -> str:
    """OpenAI API를 사용하여 LLM에 질문하고 응답을 받습니다."""
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
        # 🚨🚨 LLM 통신 실패 시 오류를 강제로 출력하는 핵심 라인 🚨🚨
        # type(e).__name__을 사용하여 오류 클래스 이름(예: AuthenticationError)을 출력합니다.
        print(f"[CRITICAL-LLM] ❌ LLM 통신 중 치명적인 오류 발생: {type(e).__name__}: {e}")
        return "죄송합니다. 서버가 인공지능과 통신할 수 없습니다."

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
        ensure_db_connection() # DB 연결 상태 다시 확인
        
        # 1. 이벤트 로그 가져오기 (events 테이블)
        sql_events = """
            SELECT ts, module, action, payload
            FROM events
            WHERE ts >= UTC_TIMESTAMP() - INTERVAL %s MINUTE
            ORDER BY ts ASC
        """
        CURSOR.execute(sql_events, (minutes,)) 
        rows = CURSOR.fetchall()
        logs = [f"[{r[0]}] ({r[1]}) {r[2]} → {r[3]}" for r in rows]
        print(f"[{now_str()}] [DB] Retrieved {len(logs)} event logs.")

        # 2. IMU 통계 가져오기 (imu_data 테이블)
        # 최대/최소 Roll (기울기)
        sql_roll = """
            SELECT MAX(roll), MIN(roll)
            FROM imu_data
            WHERE ts >= UTC_TIMESTAMP() - INTERVAL %s MINUTE
        """
        CURSOR.execute(sql_roll, (minutes,))
        max_roll, min_roll = CURSOR.fetchone()
        imu_stats['max_roll'] = max_roll if max_roll is not None else 0.0
        imu_stats['min_roll'] = min_roll if min_roll is not None else 0.0

        # 최신 Yaw (현재 방향)
        sql_yaw = """
            SELECT yaw
            FROM imu_data
            WHERE ts >= UTC_TIMESTAMP() - INTERVAL %s MINUTE
            ORDER BY ts DESC 
            LIMIT 1
        """
        CURSOR.execute(sql_yaw, (minutes,))
        latest_yaw_result = CURSOR.fetchone()
        imu_stats['latest_yaw'] = latest_yaw_result[0] if latest_yaw_result else 0.0
        
        print(f"[{now_str()}] [DB] Retrieved IMU statistics.")
        
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

    # 🚨 재생 시작 상태를 GUI에 알림 🚨
    try:
        publish.single(STATUS_TOPIC, payload="TTS_START", hostname=BROKER, auth={'username': MQTT_USERNAME, 'password': MQTT_PASSWORD})
    except Exception as e:
        print(f"[WARN] Failed to publish TTS_START status: {e}")

    with TTS_LOCK:
        if TTS_PROCESS and TTS_PROCESS.poll() is None:
            # 기존 TTS 중단
            TTS_PROCESS.terminate()
            TTS_PROCESS.wait()
        try:
            clean_text = clean_tts_text(text)
            tts = gTTS(text=clean_text, lang="ko")
            tts.save(filename)
            TTS_PROCESS = subprocess.Popen(["mpv", "--no-terminal", "--volume=100", "--speed=1.3", filename])
        except Exception as e:
            print(f"[TTS Error] {e}")

# =======================================================================
# === [STT/음성 명령] 스레드 로직 추가 ===
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
    
    # 🚨 STT 안정화 수정 1: 응답 시간 및 구문 시간 제한 확대
    # Google API로부터 응답을 기다리는 최대 시간 (네트워크 지연 대비)
    r.operation_timeout = 5 
    # 인식기가 구문이 끝났다고 판단하기 전까지의 최대 시간 (긴 문장 대비)
    # r.pause_threshold = 0.8 (기본값)
    
    mqtt_broker = BROKER
    auth_data = {'username': MQTT_USERNAME, 'password': MQTT_PASSWORD}
    
    # ----------------------------------------------------------------------
    # TODO: [사용자 지정] 여기에 STT를 시도할 마이크 장치 인덱스를 넣어보세요.
    # ----------------------------------------------------------------------
    DEVICE_INDEX = None # 기본값: 시스템 기본 마이크 사용

    # 마이크 스트림을 루프 바깥에서 한 번만 엽니다 (효율성 및 안정성 개선)
    try:
        # 마이크 스트림을 루프 바깥에서 한 번만 열어 효율성을 개선합니다.
        with sr.Microphone(device_index=DEVICE_INDEX, sample_rate=16000) as source:
            print("[STT-THREAD] Ambient noise calibrating...")
            r.adjust_for_ambient_noise(source, duration=1.5)
            print("[STT-THREAD] Setup complete. Starting speech recognition loop...")
            
            # 메인 리스닝 루프
            while True:
                # 🚨 STT 안정화 수정 2: 구문 시간 제한을 15초로 늘려 긴 명령 인식 보장
                print("\n[STT-THREAD] Listening for command (Say '최근 N분 요약해줘')...")
                audio = r.listen(source, timeout=None, phrase_time_limit=15) 
                
                print("[STT-THREAD] Recognizing speech...")
                
                try:
                    text = r.recognize_google(audio, language="ko-KR") 
                    print("[STT-THREAD] You said:", text)

                    # TTS Stop Logic
                    stop_keywords = ["그만", "멈춰", "중단", "정지", "닥쳐"]
                    if any(keyword in text for keyword in stop_keywords):
                         with TTS_LOCK:
                            if TTS_PROCESS and TTS_PROCESS.poll() is None:
                                TTS_PROCESS.terminate()
                                TTS_PROCESS.wait()
                                print("[STT-THREAD] 🛑 TTS playback terminated by voice command.")
                                continue 

                    topic, payload = parse_speech_command(text)
                    
                    # MQTT 전송
                    try:
                        publish.single(topic,
                                       payload=payload,
                                       hostname=mqtt_broker,
                                       qos=1,
                                       auth=auth_data)
                        print(f"[STT-THREAD] MQTT Published: {topic} -> {payload}")
                    except Exception as e:
                        print(f"[STT-THREAD] MQTT publish error: {e}")
                        
                # 🚨 STT 안정화 수정 3: UnknownValueError와 RequestError 분리 처리
                except sr.UnknownValueError:
                    # 마이크에 소리가 있었으나, Google이 텍스트로 변환하지 못한 경우
                    print("[STT-THREAD] ⚠️ Recognition Failed: Google Speech Recognition could not understand audio. (Please speak louder or clearer.)")
                except sr.RequestError as e:
                    # 네트워크 문제, API 키 문제 등 Google 서비스에 요청 실패한 경우
                    print(f"[STT-THREAD] ❌ Request Error: Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    # 기타 예상치 못한 오류
                    print(f"[STT-THREAD] ❗ Unexpected Error during recognition: {e}")

                time.sleep(0.1) # 루프 안정화

    except Exception as e:
        # 초기화 실패 또는 루프 내부의 예상치 못한 치명적 오류 (e.g., 오디오 장치 유실)
        print(f"[CRITICAL] STT Loop or Initialization Error: {e}")
        time.sleep(1) # 오류 발생 시 잠시 대기 후 종료

# === MQTT 콜백 함수 (메인 로직) ===
def on_connect(client, userdata, flags, rc):
    """브로커 연결 시 호출되며, 토픽을 구독합니다."""
    if rc == 0:
        print("[OK] Connected to broker")
        # TOPIC_BASE와 COMMAND_TOPIC을 사용하여 구독
        client.subscribe(TOPIC_BASE + "#") 
        client.subscribe("command/#") # 모든 command/ 토픽 구독 (summary, query 포함)
        print(f"[{now_str()}] [SUB] Subscribed to {TOPIC_BASE}# and command/#")
    else:
        print(f"[{now_str()}] [FAIL] Connection failed, code: {rc}")

# === [데이터 라우터] 핵심 로직 ===

def process_and_save_data(msg):
    """
    수신된 MQTT 메시지를 분석하여 알맞은 테이블에 저장하고,
    필요 시 이벤트를 생성합니다.
    """

    # 1. 토픽 파싱
    topic = msg.topic
    payload = msg.payload.decode('utf-8', errors='ignore')
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
    # 🎥 VIDEO 토픽은 frames 테이블로만 저장 (events에는 저장 안 함)
    # =======================================================
    if action == "VIDEO":
        save_frame_data(module, payload)
        print(f"[{now_str()}] [FRAME] 🖼 Saved {module} frame ({len(payload):,} bytes)")
        return  # ✅ VIDEO는 여기서 종료 (events로 안 감)

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
        return

    # 2-2. 🟢 RAW 토픽 처리 (INFO 레벨 - 연속 데이터)
    elif action == "RAW":
        if module == "IMU":
            save_imu_raw_data(payload_dict)
            print(f"[{now_str()}] [DB] Saved IMU RAW data to imu_data table.")

        # VISION 시스템의 모든 세부 모듈(AD, PE 포함) 데이터를 vision_data에 통합 저장합니다.
        elif module in ["VISION", "AD", "PE"]:
            save_vision_data(module, action, payload_dict)
            print(f"[{now_str()}] [DB] Saved {module} RAW data to vision_data table.")

        else:
            print(f"[{now_str()}] [WARN] Unknown RAW module: {module}. Data discarded.")
        return

    # 2-3. 기타 일반 시스템/STT 이벤트 (events 테이블)
    else:
        save_event_log(module, action, payload)
        print(f"[{now_str()}] [LOG] Saved general log to events table. Module: {module}")
        
# === [MQTT 콜백] 명령어 처리 후 데이터 라우팅을 'process_and_save_data'로 위임하는 진입점. ===
def on_message(client, userdata, msg):
    """메시지가 수신될 때 호출되며, 토픽에 따라 데이터 저장 또는 명령을 처리합니다."""
    now = now_str() 
    payload = msg.payload.decode()
    topic = msg.topic
    print(f"[{now}] {topic} → {payload}") 

    # 1. === 명령어/요약 트리거 처리 (동적 시간 파싱) ===
    if topic.startswith("command/"):
        
        if topic == "command/summary":
            print(f"[{now}] [CMD] Summary request received → Generating report...")
            
            # Summary 요청이 들어왔음을 이벤트 로그에 기록
            minutes_payload = payload.strip()
            save_event_log("STT", "SUMMARY_REQUEST", f"Request for summary (Payload: {minutes_payload})")

            minutes = 15 # 기본값은 15분
            try:
                # payload는 '30'과 같은 문자열 분 단위이거나 'minutes=30' 형태
                minutes = int(minutes_payload)
            except ValueError:
                pass # payload가 단순 숫자가 아닐 경우 무시하고 기본값 15분 유지
            
            # 최소 1분 이상, 최대 180분(3시간)까지만 처리하도록 제한 (안전성 확보)
            minutes = max(1, min(minutes, 180)) 

            print(f"[{now}] Fetching logs for the last {minutes} minutes.")
            logs, imu_stats = fetch_logs(minutes) 
            
            summary = summarize_logs(logs, imu_stats, minutes) 
            text_to_speech(summary)
            
            # 이곳에 LLM 보고서 저장 함수를 추가합니다.
            save_llm_report(summary)

            # LLM 결과 TTS 발화 후 DB에 기록
            save_event_log("LLM", "SAY", summary)

        elif topic == "command/query":
             # 일반 쿼리는 LLM에 바로 질의 후 답변을 TTS로 발화합니다.
             print(f"[{now}] [CMD] Query request received → {payload}")
             # 사용자 쿼리를 이벤트 로그에 기록
             save_event_log("STT", "QUERY", payload)
             
             # LLM 질의
             response = query_llm(payload)
             text_to_speech(response)
             # LLM 답변을 이벤트 로그에 기록
             save_event_log("LLM", "RESPONSE", response)

        return

    # 2. === 데이터 처리 로직을 새로운 함수로 위임 ===
    process_and_save_data(msg)
    

# === MQTT 클라이언트 및 메인 루프 ===
# MQTTv311 프로토콜 명시로 DeprecationWarning 해결
client = mqtt.Client(client_id="MarineServer", protocol=mqtt.MQTTv311) 

# MQTT 인증 정보 설정
client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

# === 서버 실행 및 루프 ===
# 메인 루프를 try 블록으로 감싸서 종료 시 DB/MQTT 자원 정리 보장
try:
    # 1. 브로커 연결
    print("[INFO] Connecting to broker...")
    client.connect(BROKER, PORT, 60)

    # 2. STT/TTS 기능 테스트 및 스레드 시작
    stt_recognizer = sr.Recognizer()
    microphone_test_result = check_microphone(stt_recognizer)
    
    if microphone_test_result:
        stt_thread = threading.Thread(target=stt_listening_loop)
        stt_thread.daemon = True # 메인 스레드 종료 시 함께 종료
        stt_thread.start()
        print("[INFO] STT Listening Thread started.")
    else:
        print("\n[WARN] 마이크 초기화 실패로 STT/TTS 기능 스레드는 시작되지 않았습니다.")

    # 3. 스피커 테스트 (TTS 기능 확인)
    check_speaker()
    
    # 4. 메인 MQTT 루프 실행 (STT와 동시 실행)
    print("[INFO] Server is running. Entering MQTT loop_forever(). Press Ctrl+C to stop.")
    client.loop_forever()
    
except KeyboardInterrupt:
    # Ctrl+C가 눌렸을 때 깔끔하게 종료
    print("\n[EXIT] Server is stopping gracefully (KeyboardInterrupt)...")
except Exception as e:
    # 예상치 못한 치명적 오류 처리 (예: MQTT 연결 실패, 초기화 오류 등)
    print(f"\n[CRITICAL-ERROR] Server stopped due to unexpected error: {e}")

finally:
    # 5. 자원 정리 (정상 종료, 키보드 인터럽트, 또는 치명적 오류 발생 시 모두 실행)
    print("[EXIT] Cleaning up resources...")
    client.disconnect()
    
    # 전역 변수가 정의되어 있는지 확인 후 닫습니다.
    if 'CURSOR' in globals() and CURSOR:
        CURSOR.close() 
    if 'DB_CONN' in globals() and DB_CONN:
        DB_CONN.close()
    
    print("[EXIT] Server stopped successfully.")
