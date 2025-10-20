import paho.mqtt.client as mqtt
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import threading
import json
import time
import base64
import numpy as np
import cv2
from datetime import datetime
import pymysql

# ====================================================
# 1. 시스템 설정
# ====================================================
# 🚨 사용자 환경에 맞게 DB/MQTT 설정 변경 필요 🚨
DB_HOST = "localhost" # DB가 이 서버(RPi 5)에 있으므로
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# MQTT 브로커 설정 (같은 RPi 5에서 브로커가 실행 중인 경우)
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883

# 대시보드에서 구독할 토픽
IMU_TOPIC = "project/imu/RAW"
AD_VIDEO_TOPIC = "project/vision/AD/VIDEO" # 클라이언트에서 이 토픽으로 Base64 프레임 발행 필요
PE_VIDEO_TOPIC = "project/vision/PE/VIDEO" # 클라이언트에서 이 토픽으로 Base64 프레임 발행 필요
LOG_TOPIC_SUBSCRIPTION = "project/#" # 모든 project/* 로그 수신

# ====================================================
# 2. 전역 변수 및 데이터 저장소
# ====================================================
# 실시간 데이터 저장
latest_imu_data = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
latest_ad_frame = b'' # Base64로 인코딩된 JPEG 바이트 스트림 저장
latest_pe_frame = b''
log_buffer = [] # 최근 로그 100개 저장

# 락 및 Flask 초기화
ad_lock = threading.Lock()
pe_lock = threading.Lock()
imu_lock = threading.Lock()
log_lock = threading.Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key' # 보안을 위해 실제 키로 변경
socketio = SocketIO(app, async_mode='eventlet') # eventlet으로 설정

# ====================================================
# 3. MQTT 클라이언트 로직
# ====================================================

def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] Connected with result code {rc}. Subscribing to topics...")
    # 모든 관련 토픽 구독
    client.subscribe([(IMU_TOPIC, 0), (AD_VIDEO_TOPIC, 0), (PE_VIDEO_TOPIC, 0), (LOG_TOPIC_SUBSCRIPTION, 0)])

def on_message(client, userdata, msg):
    global latest_imu_data, latest_ad_frame, latest_pe_frame, log_buffer
    
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    
    # --- 1. IMU 데이터 처리 ---
    if topic == IMU_TOPIC:
        try:
            data = json.loads(payload)
            with imu_lock:
                latest_imu_data = {
                    "roll": round(data.get("roll", 0.0), 2),
                    "pitch": round(data.get("pitch", 0.0), 2),
                    "yaw": round(data.get("yaw", 0.0), 2)
                }
            # 웹 클라이언트에 실시간으로 IMU 데이터 전송
            socketio.emit('imu_update', latest_imu_data)
        except json.JSONDecodeError:
            print(f"[ERROR] IMU data JSON decode failed: {payload}")
    
    # --- 2. 비디오 프레임 처리 (Base64) ---
    elif topic == AD_VIDEO_TOPIC:
        # Base64 문자열을 바이트로 저장
        with ad_lock:
            latest_ad_frame = msg.payload
    elif topic == PE_VIDEO_TOPIC:
        with pe_lock:
            latest_pe_frame = msg.payload

    # --- 3. 실시간 로그 처리 ---
    if topic.startswith('project/') and topic not in [IMU_TOPIC, AD_VIDEO_TOPIC, PE_VIDEO_TOPIC]:
        try:
            # 로그 형식 통일: [YYYY-MM-DD HH:MM:SS] [LEVEL] [TOPIC] MSG
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # JSON 형태로 들어온 로그 메시지를 파싱하여 메시지만 추출 시도
            try:
                data = json.loads(payload)
                module = data.get('module', 'UNKNOWN')
                level = data.get('level', 'INFO')
                message = f"{module} - {data.get('description', data.get('detections', 'No description'))}"
            except json.JSONDecodeError:
                # JSON이 아니면 RAW 페이로드 자체를 메시지로 사용
                level = 'INFO'
                message = payload[:100] # 너무 길면 잘라냄

            log_entry = f"[{log_time}] [{level.upper()}] [{topic.split('/')[-1]}] {message}"
            
            with log_lock:
                log_buffer.append(log_entry)
                # 로그 버퍼 크기 유지
                if len(log_buffer) > 100:
                    log_buffer.pop(0)
            
            # 웹 클라이언트에 로그 항목 전송
            socketio.emit('log_message', {'log': log_entry})
            
        except Exception as e:
            print(f"[ERROR] Log message processing failed: {e}")

# MQTT 스레드 함수
def mqtt_thread_function():
    client = mqtt.Client(client_id="DashboardClient")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"[CRITICAL] MQTT thread connection failed: {e}")
        time.sleep(5)

# ====================================================
# 4. Flask 라우트 (Route)
# ====================================================

# 메인 페이지 라우트
@app.route('/')
def index():
    # 시작 시 DB에서 최근 50개의 로그를 불러와 초기 로그로 사용
    initial_db_logs = fetch_db_logs(50)
    return render_template('index.html', initial_db_logs=initial_db_logs)

# 비디오 스트리밍 함수 (각 카메라 피드에서 호출)
def generate_frame(latest_frame_ref, lock):
    """MJPEG 스트림을 생성하는 제너레이터 함수"""
    while True:
        with lock:
            frame_data = latest_frame_ref
        
        if frame_data:
            try:
                # Base64 디코드 -> JPEG 바이트 배열 -> OpenCV Mat 객체
                np_array = np.frombuffer(base64.b64decode(frame_data), np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                # JPEG 인코딩 (웹 스트리밍용)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # MJPEG 프레임 반환
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            except Exception as e:
                # 프레임 디코딩/인코딩 오류 처리
                print(f"[ERROR] Video stream generation failed: {e}")
                time.sleep(0.1) # 오류 발생 시 짧게 대기하여 루프 폭주 방지
                
        time.sleep(1/30) # 30 FPS 제한

# AD 카메라 비디오 피드 라우트
@app.route('/video_feed/AD')
def video_feed_ad():
    return Response(generate_frame(latest_ad_frame, ad_lock), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# PE 카메라 비디오 피드 라우트
@app.route('/video_feed/PE')
def video_feed_pe():
    return Response(generate_frame(latest_pe_frame, pe_lock), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ====================================================
# 5. DB 유틸리티
# ====================================================

def fetch_db_logs(limit=50):
    """DB의 events 테이블에서 최근 로그를 가져옵니다."""
    try:
        db = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, 
                             database=DB_NAME, charset='utf8mb4')
        cursor = db.cursor()
        
        # marine_system_backup.sql에 있는 events 테이블의 데이터를 가져옵니다.
        # ts: timestamp, module: string, level: string, description: string
        query = f"SELECT ts, module, level, description FROM events ORDER BY ts DESC LIMIT {limit}"
        cursor.execute(query)
        
        results = []
        for row in cursor.fetchall():
            ts, module, level, description = row
            # 로그 형식 통일: [YYYY-MM-DD HH:MM:SS] [LEVEL] [MODULE] DESC
            log_time = ts.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{log_time}] [{level.upper()}] [{module}] {description}"
            results.append(log_entry)
            
        cursor.close()
        db.close()
        # 최신 로그가 맨 아래로 오도록 순서 반전
        return results[::-1]
    except Exception as e:
        print(f"[DB-ERROR] Failed to fetch initial logs: {e}")
        return [f"[DB-ERROR] 초기 DB 로그 로드 실패: {e}"]

# ====================================================
# 6. 서버 실행
# ====================================================

if __name__ == '__main__':
    # MQTT 클라이언트 스레드 시작
    mqtt_thread = threading.Thread(target=mqtt_thread_function, daemon=True)
    mqtt_thread.start()
    
    # Flask 서버 시작
    print(f"\n[INFO] Starting Flask-SocketIO server on http://{MQTT_BROKER}:5000")
    # 0.0.0.0으로 바인딩하여 외부 접속 허용
    socketio.run(app, host='0.0.0.0', port=5000)
