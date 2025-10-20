import sys
import threading
import time
import json
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pymysql
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QSplitter, QTextEdit, QLabel, QGridLayout
)
# PySide6에서는 Qt와 QObject를 QtCore에서 임포트합니다.
from PySide6.QtCore import Qt, QTimer, QObject, Signal as pyqtSignal
from PySide6.QtGui import QImage, QPixmap

# =========================================================
# 1. 환경 설정 (server.py의 설정과 동일하게 맞춰야 함)
# =========================================================

# DB 설정 (현재 서버에서 실행)
DB_HOST = "localhost"
DB_USER = "marine_user"
DB_PASSWORD = "sksk"
DB_NAME = "marine_system"

# MQTT 설정 (server.py의 BROKER가 "0.0.0.0"이면 localhost 사용)
MQTT_BROKER = "127.0.0.1" 
MQTT_PORT = 1883
IMU_RAW_TOPIC = "project/imu/RAW"
EVENTS_FETCH_INTERVAL_MS = 1000 # DB 로그 조회 주기 (1초)

# 카메라 스트림 URL (실제 환경에 맞게 수정 필요)
# AD 클라이언트 (젯슨 나노)가 RTSP/RTMP로 스트리밍하고 있어야 합니다.
AD_STREAM_URL = "rtsp://10.10.14.73/ad_stream" # 예시 URL
# PE 클라이언트 (라즈베리파이 5)가 RTSP/RTMP로 스트리밍하고 있어야 합니다.
PE_STREAM_URL = "rtsp://10.10.14.73/pe_stream" # 예시 URL
# 참고: FFmpeg/OpenCV가 지원하는 RTMP, RTSP, 또는 HTTP 스트림을 사용하세요.


# =========================================================
# 2. PyQt 신호 객체 (GUI 업데이트를 위한 스레드 안전 메커니즘)
# =========================================================
class GuiUpdater(QObject):
    # IMU 데이터 수신 시 업데이트할 시그널
    imu_data_signal = pyqtSignal(dict) 
    # DB 로그 수신 시 업데이트할 시그널
    db_log_signal = pyqtSignal(str) 
    # 영상 프레임 수신 시 업데이트할 시그널 (AD/PE, 프레임 데이터)
    frame_signal = pyqtSignal(str, np.ndarray) 

# =========================================================
# 3. 비디오 캡처 스레드 클래스
# =========================================================
class VideoCaptureThread(threading.Thread):
    def __init__(self, stream_url, module_name, updater):
        super().__init__()
        self.stream_url = stream_url
        self.module_name = module_name
        self.updater = updater
        self._stop_event = threading.Event()

    def run(self):
        # cv2.CAP_FFMPEG을 사용하여 RTSP/RTMP 스트림의 안정성 확보 (Linux 환경)
        cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[ERROR] {self.module_name} 스트림 연결 실패: {self.stream_url}")
            self.updater.db_log_signal.emit(f"⚠️ {self.module_name} 영상 스트림 연결 실패.")
            return

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"[WARN] {self.module_name} 스트림 재연결 시도...")
                cap.release()
                cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
                time.sleep(1)
                continue

            # GUI에 프레임 전송 (QImage 변환은 메인 스레드에서 수행)
            self.updater.frame_signal.emit(self.module_name, frame)
            
            # CPU 점유율 조절
            time.sleep(1/30) # 초당 30프레임 목표

        cap.release()

    def stop(self):
        self._stop_event.set()

# =========================================================
# 4. DB 로그 조회 스레드 클래스
# =========================================================
class DBLogFetcher(QObject):
    def __init__(self, updater):
        super().__init__()
        self.updater = updater
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fetch_logs)
        self.timer.start(EVENTS_FETCH_INTERVAL_MS) # 1초마다 조회
        self.last_id = 0

    def get_db_connection(self):
        """DB 연결 객체를 생성하고 반환합니다."""
        try:
            db = pymysql.connect(
                host=DB_HOST, user=DB_USER, password=DB_PASSWORD, 
                database=DB_NAME, charset="utf8mb4"
            )
            return db
        except Exception as e:
            print(f"[DB-ERROR] 연결 실패: {e}")
            return None

    def _fetch_logs(self):
        db = self.get_db_connection()
        if not db:
            return

        try:
            with db.cursor() as cursor:
                # events 테이블에서 마지막으로 읽은 id 이후의 새로운 로그를 가져옵니다.
                sql = "SELECT id, ts, module, action, payload FROM events WHERE id > %s ORDER BY id ASC"
                cursor.execute(sql, (self.last_id,))
                rows = cursor.fetchall()

                if rows:
                    log_messages = []
                    for row in rows:
                        self.last_id = row[0] # 마지막 ID 업데이트
                        ts, module, action, payload = row[1:]
                        # 로그 메시지 포맷팅
                        log_msg = f"[{ts}] ({module}/{action}) {payload}"
                        log_messages.append(log_msg)
                    
                    # 새로운 로그가 있을 때만 GUI에 전송
                    self.updater.db_log_signal.emit("\n".join(log_messages))
                    
        except Exception as e:
            print(f"[DB-ERROR] 로그 조회 실패: {e}")
        finally:
            if db:
                db.close()

# =========================================================
# 5. 메인 대시보드 GUI 클래스
# =========================================================
class MarineDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("해양 시스템 통합 모니터링")
        self.setGeometry(100, 100, 1600, 900)
        
        # GUI 업데이트를 위한 시그널 객체 초기화
        self.updater = GuiUpdater()
        self.updater.imu_data_signal.connect(self.update_imu_data)
        self.updater.db_log_signal.connect(self.update_db_log)
        self.updater.frame_signal.connect(self.update_video_frame)
        
        # MQTT 클라이언트 초기화 및 연결
        self._init_mqtt()
        
        # GUI 위젯 초기화 및 레이아웃 설정
        self._init_ui()

        # DB 로그 패처 초기화
        self.db_fetcher = DBLogFetcher(self.updater)

        # 비디오 스레드 초기화 및 시작
        self._init_video_streams()

    def _init_mqtt(self):
        self.client = mqtt.Client()
        self.client.on_connect = self._on_mqtt_connect
        self.client.on_message = self._on_mqtt_message
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start() 
            print("[INFO] MQTT Client connected and loop started.")
        except Exception as e:
            print(f"[CRITICAL] MQTT 연결 실패: {e}")
            sys.exit(1)

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(IMU_RAW_TOPIC)
            print(f"[INFO] Subscribed to {IMU_RAW_TOPIC}")
        else:
            print(f"[ERROR] MQTT Connection failed (RC: {rc})")

    def _on_mqtt_message(self, client, userdata, msg):
        # IMU 데이터 수신 처리
        if msg.topic == IMU_RAW_TOPIC:
            try:
                payload_dict = json.loads(msg.payload.decode('utf-8'))
                # 시그널을 통해 메인 스레드로 데이터 전송
                self.updater.imu_data_signal.emit(payload_dict)
            except json.JSONDecodeError:
                print("[ERROR] IMU Payload JSON decode failed.")
            except Exception as e:
                print(f"[ERROR] IMU Message processing error: {e}")

    def _init_ui(self):
        # 메인 레이아웃: 세로로 반 분할
        main_layout = QHBoxLayout(self)
        
        # 1. 왼쪽 패널: DB 로그
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setWindowTitle("DB 이벤트 및 시스템 로그")
        self.log_widget.setStyleSheet("font-family: Consolas; font-size: 10pt; color: #E0E0E0; background-color: #2D2D30;")
        
        # 2. 오른쪽 패널: IMU + 영상 2개 (세로 배치)
        right_panel = QWidget()
        right_vbox = QVBoxLayout(right_panel)

        # 2-1. IMU 데이터 출력
        self.imu_label = QLabel("IMU Data: 대기 중...")
        self.imu_label.setAlignment(Qt.AlignCenter)
        self.imu_label.setStyleSheet("font-size: 16pt; font-weight: bold; padding: 10px; border: 1px solid #606060; background-color: #3C3C3C;")
        right_vbox.addWidget(self.imu_label)

        # 2-2. 영상 2개 (Grid)
        video_grid = QWidget()
        grid_layout = QGridLayout(video_grid)
        
        # AD 카메라 영상
        self.ad_video_label = QLabel("AD Camera (영상 대기 중)")
        self.ad_video_label.setAlignment(Qt.AlignCenter)
        self.ad_video_label.setStyleSheet("border: 2px solid #FF5733; background-color: black;")
        grid_layout.addWidget(self.ad_video_label, 0, 0)
        
        # PE 카메라 영상
        self.pe_video_label = QLabel("PE Camera (영상 대기 중)")
        self.pe_video_label.setAlignment(Qt.AlignCenter)
        self.pe_video_label.setStyleSheet("border: 2px solid #33FF57; background-color: black;")
        grid_layout.addWidget(self.pe_video_label, 1, 0)

        right_vbox.addWidget(video_grid)

        # 3. QSplitter를 사용하여 왼쪽/오른쪽 크기 조절 가능하게 분할
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.log_widget)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1100]) # 초기 분할 비율 설정

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _init_video_streams(self):
        # AD 영상 스트림 스레드 시작
        self.ad_thread = VideoCaptureThread(AD_STREAM_URL, "AD", self.updater)
        self.ad_thread.start()

        # PE 영상 스트림 스레드 시작
        self.pe_thread = VideoCaptureThread(PE_STREAM_URL, "PE", self.updater)
        self.pe_thread.start()

    # =========================================================
    # 6. 슬롯 함수 (GUI 업데이트)
    # =========================================================
    def update_imu_data(self, data: dict):
        """IMU 데이터 수신 시 라벨 업데이트"""
        roll = data.get('roll', 0.0)
        pitch = data.get('pitch', 0.0)
        yaw = data.get('yaw', 0.0)
        
        text = f"Roll: {roll:6.2f}° | Pitch: {pitch:6.2f}° | Yaw: {yaw:6.2f}°"
        
        # 롤 각도에 따른 색상 변화 (위험 경고)
        if abs(roll) > 25:
            color = "#FF3333" # 빨강
        elif abs(roll) > 10:
            color = "#FFD700" # 노랑
        else:
            color = "#33FF57" # 초록
            
        self.imu_label.setStyleSheet(
            f"font-size: 16pt; font-weight: bold; padding: 10px; border: 1px solid #606060; background-color: #3C3C3C; color: {color};"
        )
        self.imu_label.setText(text)

    def update_db_log(self, log_message: str):
        """DB 로그 수신 시 QTextEdit에 추가"""
        # 스크롤을 맨 아래로 내리면서 텍스트 추가
        self.log_widget.append(log_message)

    def update_video_frame(self, module_name: str, frame: np.ndarray):
        """비디오 프레임 수신 시 라벨 업데이트"""
        
        # OpenCV BGR 프레임을 PyQt RGB 프레임으로 변환
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 라벨 크기에 맞게 이미지 스케일 조정
        if module_name == "AD":
            label = self.ad_video_label
        elif module_name == "PE":
            label = self.pe_video_label
        else:
            return

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        label.setText("") # 영상이 출력되면 '대기 중' 텍스트 제거
        

    def closeEvent(self, event):
        """창 종료 시 스레드 종료 및 MQTT 연결 해제"""
        print("[INFO] GUI closing. Stopping threads...")
        self.ad_thread.stop()
        self.pe_thread.stop()
        self.ad_thread.wait()
        self.pe_thread.wait()
        self.client.loop_stop()
        self.client.disconnect()
        self.db_fetcher.timer.stop()
        event.accept()
    
# =========================================================
# 7. 메인 실행 (수정된 부분)
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 전체 스타일을 어둡게 설정 (선박 모니터링 환경에 적합)
    app.setStyle('Fusion')
    palette = app.palette()

    # Qt.GlobalColor을 사용하기 위해 임포트 (기존 임포트 유지)
    from PySide6.QtCore import Qt as GlobalQt 

    # 🚨🚨 PySide6 호환을 위해 QPalette 속성 이름(Role)을 소문자(window)로 변경 🚨🚨
    palette.setColor(palette.Window, GlobalQt.GlobalColor.darkGray)       # Window -> window
    palette.setColor(palette.WindowText, GlobalQt.GlobalColor.white)     # WindowText -> windowText
    palette.setColor(palette.Base, GlobalQt.GlobalColor.black)           # Base -> base
    palette.setColor(palette.AlternateBase, GlobalQt.GlobalColor.darkGray) # AlternateBase -> alternateBase
    palette.setColor(palette.ToolTipBase, GlobalQt.GlobalColor.white)    # ToolTipBase -> toolTipBase
    palette.setColor(palette.ToolTipText, GlobalQt.GlobalColor.white)    # ToolTipText -> toolTipText
    palette.setColor(palette.Text, GlobalQt.GlobalColor.white)           # Text -> text
    palette.setColor(palette.Button, GlobalQt.GlobalColor.darkGray)      # Button -> button
    palette.setColor(palette.ButtonText, GlobalQt.GlobalColor.white)     # ButtonText -> buttonText
    palette.setColor(palette.BrightText, GlobalQt.GlobalColor.red)       # BrightText -> brightText
    palette.setColor(palette.Link, GlobalQt.GlobalColor.cyan)            # Link -> link
    palette.setColor(palette.Highlight, GlobalQt.GlobalColor.blue)       # Highlight -> highlight
    palette.setColor(palette.HighlightedText, GlobalQt.GlobalColor.black) # HighlightedText -> highlightedText

    app.setPalette(palette)

    # GUI 실행
    dashboard = MarineDashboard()
    dashboard.show()
    sys.exit(app.exec())