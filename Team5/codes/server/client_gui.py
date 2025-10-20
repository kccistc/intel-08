import sys
import os
import json
import base64
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QGroupBox, QLabel, QTextEdit, 
    QGridLayout, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QTabWidget
)
from PyQt6.QtGui import QFont, QFontDatabase, QImage, QPixmap, QRegion
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QSize, QTimer, QRect

import paho.mqtt.client as mqtt

# --- Global Configuration ---
MQTT_BROKER = "10.10.14.73"
MQTT_PORT = 1883

TOPIC_BASE = "project/vision"
MQTT_USERNAME = "PYQT_USER"
MQTT_PASSWORD = "sksk"

TOPIC_IMU = "project/imu/RAW"
TOPIC_CAM_PE = f"{TOPIC_BASE}/FALL/VIDEO"   # PE.py의 비디오 스트림
TOPIC_PE_RAW = f"{TOPIC_BASE}/PE/RAW"       # 낙상 감지 RAW 로그
TOPIC_PE_ALERT = f"{TOPIC_BASE}/PE/ALERT"   # 낙상 감지 ALERT 로그
TOPIC_CAM_AD = f"{TOPIC_BASE}/AD/RAW"
TOPIC_VIDEO_AD = f"{TOPIC_BASE}/AD/VIDEO"
TOPIC_LOGS = f"project/log/RAW"
TOPIC_LOGBOOK = "project/log/LOGBOOK"

def safe_b64decode(data: str):
    data = data.strip().replace('\n', '').replace('\r', '')
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    try:
        return base64.b64decode(data)
    except Exception as e:
        print(f"[Decode Error] {e}")
        return b''

COLOR_MAP = {
    "IMU": "#58a6ff",
    "AD": "#e76f51",
    "PE": "#9d4edd",
    "SERVER": "#2a9d8f",
    "STT": "#2a9d8f",
    "LLM": "#2a9d8f",
    "DEFAULT": "#a8a8a8"
}

# --- MQTT Client ---
class MqttClient(QObject):
    message_signal = pyqtSignal(str, str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = mqtt.Client(client_id="PYQT_Dashboard_Client")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.username_pw_set(username=MQTT_USERNAME, password=MQTT_PASSWORD)

    def on_connect(self, client, userdata, flags, rc):
        client.subscribe(TOPIC_VIDEO_AD)
        if rc == 0:
            print("MQTT Broker Connected Successfully.")
            # client.subscribe(TOPIC_IMU)
            client.subscribe(TOPIC_CAM_AD)
            client.subscribe(TOPIC_VIDEO_AD)
            client.subscribe(TOPIC_CAM_PE)     # FALL/VIDEO
            client.subscribe(TOPIC_PE_RAW)     # 낙상 RAW
            client.subscribe(TOPIC_PE_ALERT)   # 낙상 ALERT
            client.subscribe(TOPIC_LOGS)
            client.subscribe(TOPIC_LOGBOOK)
            print(f"Subscribed → {TOPIC_IMU}, {TOPIC_VIDEO_AD}, {TOPIC_CAM_PE}, {TOPIC_PE_RAW}, {TOPIC_PE_ALERT}, {TOPIC_LOGS}")
        else:
            print(f"MQTT Connection Failed with code {rc}.")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode()
            self.message_signal.emit(topic, payload)
        except Exception as e:
            print(f"Error decoding payload for topic {topic}: {e}")

    def connect_and_loop(self, broker, port, keepalive=60):
        try:
            self.client.connect(broker, port, keepalive)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection error: {e}")

# --- Main GUI ---
class MarineDashboardApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marine Server 실시간 통합 대시보드")
        self.setMinimumSize(1200, 800)

        self.imu_data = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.imu_labels = {}

        self.alert_active = False
        self.blink_state = False
        self.alert_timer = QTimer(self)
        self.alert_timer.setInterval(500)  # 0.5초 간격 깜빡임
        self.alert_timer.timeout.connect(self._blink_alert)

        self.alert_overlay = QWidget(self)
        self.alert_overlay.setStyleSheet("background-color: transparent;")
        self.alert_overlay.setGeometry(0, 0, self.width(), self.height())
        self.alert_overlay.lower()  # 맨 아래로 (내용 가리지 않게)
        self.alert_overlay.hide()

        # 창 크기 바뀌면 오버레이 크기도 자동 조정
        self.installEventFilter(self)

        # QGraphicsScene/PixmapItem 저장용
        self.ad_scene = None
        self.ad_pixmap_item = None
        self.pe_scene = None
        self.pe_pixmap_item = None

        self.init_ui()
        self.mqtt_client = self.setup_mqtt()

    # --- UI 구성 ---
    def init_ui(self):
        font_family = "Nanum Gothic" if "Nanum Gothic" in QFontDatabase.families() else "DejaVu Sans"
        self.setFont(QFont(font_family, 10))
        main_h_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QHBoxLayout(self))
        self.layout().addWidget(main_h_splitter)

        # --- 좌측 로그 창 ---
        left_log_widget = QGroupBox("데이터 로그 보기")
        tab_widget = QTabWidget()

        # 🟦 시스템 로그 탭
        self.db_log_widget = QTextEdit()
        self.db_log_widget.setReadOnly(True)
        self.db_log_widget.setFont(QFont("Monospace", 9))
        self.db_log_widget.setStyleSheet("background-color: #0d1117; color: #58a6ff;")

        # 🟧 항해일지 탭
        self.voyage_log_widget = QTextEdit()
        self.voyage_log_widget.setReadOnly(True)
        self.voyage_log_widget.setFont(QFont("Monospace", 9))
        self.voyage_log_widget.setStyleSheet("background-color: #0d1117; color: #9d4edd;")

        # 탭 구성
        tab_widget.addTab(self.db_log_widget, "시스템 로그")
        tab_widget.addTab(self.voyage_log_widget, "최근 항해일지")

        # 그룹 박스에 추가
        left_vbox = QVBoxLayout(left_log_widget)
        left_vbox.addWidget(tab_widget)

        # 메인 스플리터에 추가
        main_h_splitter.addWidget(left_log_widget)
        main_h_splitter.setSizes([400, 800])

        # --- 우측 (IMU + 카메라) ---
        right_main = QWidget()
        right_vbox = QVBoxLayout(right_main)

        # IMU 데이터
        imu_group = QGroupBox("IMU 모듈 실시간 센서 데이터 (project/IMU/RAW)")
        imu_grid = QGridLayout()
        self._setup_imu_display(imu_grid)
        imu_group.setLayout(imu_grid)

        # 카메라 (QGraphicsView 사용)
        camera_group = QGroupBox("실시간 카메라 피드 (AD & PE)")
        camera_hbox = QHBoxLayout(camera_group)

        # AD 카메라
        self.cam_ad_view = QGraphicsView()
        self.cam_ad_view.setScene(QGraphicsScene())
        self.ad_scene = self.cam_ad_view.scene()
        self.ad_pixmap_item = QGraphicsPixmapItem()
        self.ad_scene.addItem(self.ad_pixmap_item)
        self.cam_ad_view.setStyleSheet("border: 2px solid #2a9d8f; background-color: black;")
        self.cam_ad_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.cam_ad_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_ad_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_ad_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # PE 카메라
        self.cam_pe_view = QGraphicsView()
        self.cam_pe_view.setScene(QGraphicsScene())
        self.pe_scene = self.cam_pe_view.scene()
        self.pe_pixmap_item = QGraphicsPixmapItem()
        self.pe_scene.addItem(self.pe_pixmap_item)
        self.cam_pe_view.setStyleSheet("border: 2px solid #e76f51; background-color: black;")
        self.cam_pe_view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.cam_pe_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_pe_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cam_pe_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        camera_hbox.addWidget(self.cam_ad_view)
        camera_hbox.addWidget(self.cam_pe_view)

        right_vbox.addWidget(imu_group, 4)
        right_vbox.addWidget(camera_group, 6)
        main_h_splitter.addWidget(right_main)

    # --- IMU UI ---
    def _setup_imu_display(self, grid):
        data_keys = [
            ("좌우 기울어진 각도 (Roll)", "roll", "#2a9d8f"),
            ("앞뒤 기울어진 각도 (Pitch)", "pitch", "#e9c46a"),
            ("쳐다보는 방향 (Yaw)", "yaw", "#f4a261"),
        ]

        row_idx = 0
        for col, (title, key, color) in enumerate(data_keys):
            # 제목 레이블 (1행)
            t_label = QLabel(f"<b>{title}:</b>")
            grid.addWidget(t_label, 0, col*2, alignment=Qt.AlignmentFlag.AlignRight)
            # 값 레이블 (1행)
            v_label = QLabel("0.00")
            v_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
            # 각도 값 옆에 도(°) 기호를 표시하기 위해 우측 패딩을 줄입니다.
            v_label.setStyleSheet(f"color: {color}; padding: 5px 0px 5px 5px;") 
            grid.addWidget(v_label, row_idx, col*2 + 1)
            self.imu_labels[key] = v_label
            
            # 설명 레이블 추가 (2행)
            row_idx += 1
            desc_label = QLabel("데이터 없음")
            desc_label.setFont(QFont("Arial", 10))
            # 스타일을 좀 더 잘 보이게 조정했습니다.
            desc_label.setStyleSheet(f"color: {color}; font-style: italic; padding: 2px; border: 1px solid {color}; border-radius: 3px;") 
            # 가로로 2칸을 모두 차지하도록 통합합니다.
            grid.addWidget(desc_label, row_idx, col*2, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter) 
            self.imu_labels[f'{key}_desc'] = desc_label # 'roll_desc' 등을 저장
            
            row_idx -= 1 # 다음 센서는 다시 1행으로 돌아가도록 (2칸을 사용하여 총 2줄)

    # --- MQTT 설정 ---
    def setup_mqtt(self):
        client = MqttClient(self)
        client.message_signal.connect(self.on_mqtt_message)
        client.connect_and_loop(MQTT_BROKER, MQTT_PORT)
        return client

    # --- 메시지 처리 ---
    def on_mqtt_message(self, topic, payload):
        # if topic == TOPIC_IMU:
        #     try:
        #         data = json.loads(payload)
        #         self.update_imu_ui(data)
        #     except json.JSONDecodeError:
        #         print(f"[IMU] JSON Error")

        # elif topic in [TOPIC_VIDEO_AD, TOPIC_CAM_AD]:
        #     self.update_camera_view(self.ad_pixmap_item, payload)

        # elif topic == TOPIC_CAM_PE:  # 낙상 영상
        #     self.update_camera_view(self.pe_pixmap_item, payload)
        
        # elif topic == TOPIC_LOGBOOK:  # 항해일지
        #     try:
        #         data = json.loads(payload)
        #         self.update_logbook_tab(data)
        #     except Exception as e:
        #         print(f"[LOGBOOK Error] {e}")

        # elif topic in [TOPIC_LOGS, TOPIC_PE_RAW, TOPIC_PE_ALERT, TOPIC_PE_RAW]: 
        #     try:
        #         log = json.loads(payload)
        #         self.update_log_ui(log)
        #     except json.JSONDecodeError:
        #         # JSON 형식이 아닌 일반 로그 (STT 등)도 처리할 수 있도록 보강
        #         self.update_log_ui({
        #             "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #             "module": "SYS",
        #             "action": "RAW",
        #             "payload": payload
        #         })
        if topic in [TOPIC_VIDEO_AD, TOPIC_CAM_AD]:
            self.update_camera_view(self.ad_pixmap_item, payload)

        elif topic == TOPIC_CAM_PE:  # 낙상 영상
            self.update_camera_view(self.pe_pixmap_item, payload)
        
        elif topic == TOPIC_LOGBOOK:  # 항해일지
            try:
                data = json.loads(payload)
                self.update_logbook_tab(data)
            except Exception as e:
                print(f"[LOGBOOK Error] {e}")

        # 🚨 3. TOPIC_LOGS (project/log/RAW)에서 IMU 데이터 처리 로직을 추가합니다.
        # TOPIC_PE_RAW가 중복되어 있으니 하나로 정리하고 TOPIC_LOGS와 함께 묶습니다.
        elif topic in [TOPIC_LOGS, TOPIC_PE_RAW, TOPIC_PE_ALERT]: 
            try:
                log = json.loads(payload)
                
                # 💡 IMU 데이터라면 IMU UI도 업데이트
                # 🚨🚨🚨 이 조건문이 정확해야 합니다. 🚨🚨🚨
                if log.get('module') == "IMU" and log.get('action') == "RAW":
                    # log 자체가 IMU 데이터 페이로드이므로 바로 전달
                    self.update_imu_ui(log)
                    
                # 💡 모든 로그 데이터 (IMU 포함)를 시스템 로그 창에 출력
                self.update_log_ui(log) 
                
            except json.JSONDecodeError:
                # JSON 형식이 아닌 일반 로그 (STT 등)도 처리할 수 있도록 보강
                self.update_log_ui({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "module": "SYS",
                    "action": "RAW",
                    "payload": payload
                })

    # --- IMU UI 업데이트 ---
    def update_imu_ui(self, data):
        for key in self.imu_data:
            if key in data:
                try:
                    val = float(str(data[key]))
                    self.imu_labels[key].setText(f"{val:.2f}°")
                except Exception as e:
                    print(f"[IMU Error] {key}: {e}")
                    self.imu_labels[key].setText("ERR")
            
            # 설명 필드 업데이트 (새로 추가)
            desc_key = f'{key}_desc'
            if desc_key in self.imu_labels and desc_key in data:
                # 서버에서 가공한 직관적인 설명 텍스트를 바로 표시
                self.imu_labels[desc_key].setText(str(data[desc_key]))

    # --- 로그 UI 업데이트 ---
    def update_log_ui(self, log):
        """시스템 로그 탭에 사람이 읽기 좋은 형태로 출력"""
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            module = log.get('module', 'UNKNOWN').upper()
            action = log.get('action', '').upper()
            level = log.get('level', '').upper()
            color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])

            # --- payload 처리 ---
            msg_payload = log.get('payload', '')
            if isinstance(msg_payload, str):
                try:
                    msg_payload = json.loads(msg_payload)
                except Exception:
                    pass

            # 중첩 payload 제거
            if isinstance(msg_payload, dict) and "payload" in msg_payload:
                inner = msg_payload.get("payload")
                if isinstance(inner, dict):
                    msg_payload = inner

            # --- message 추출 ---
            msg = ""
            if isinstance(msg_payload, dict):
                msg = msg_payload.get('message', '')
            elif isinstance(msg_payload, list):
                msg = f"목록 {len(msg_payload)}건 수신"
            else:
                msg = str(msg_payload) or "상태 데이터 수신 완료."
            msg = " " + msg

            module_color = COLOR_MAP.get(module, COLOR_MAP["DEFAULT"])
            base_color = "#E6E6E6"  # 전체 텍스트 기본색

            if 'AD' in module:
                module_color = "#FF6600"
                module_text = "AD"
            elif 'PE' in module:
                module_color = "#9A71CF"
                module_text = "PE"
            elif 'STT' in module:
                module_color = "#06D6A0"
                module_text = "STT"
            elif 'LLM' in module:
                module_color = "#25DA0D"
                module_text = "LLM"
            elif 'IMU' in module:
                module_color = "#25ACD4"
                module_text = "IMU"
            else:
                module_color = "#A8A8A8"
                module_text = module

            if 'CRITICAL' in level or 'ALERT' in action:
                level_color = "#FF4C4C"
                level_text = "긴급"
            elif 'WARNING' in level:
                level_color = "#FFD166"
                level_text = "주의"
            elif 'INFO' in level or 'RAW' in action:
                level_color = "#FCFCFC"
                level_text = "정보"
            else:
                level_color = "#A8A8A8"
                level_text = "안전"

            # --- 최종 출력 ---
            formatted = (
                f"<pre style='color:{base_color}; font-family:monospace;'>"
                f"[{ts}]  "
                f"<span style='color:{module_color}; font-weight:bold;'>{module:<6}</span>"
                f"<span style='color:{level_color};'>[{level_text:^4}]</span>  "
                f"{msg}</pre><br>"
            )

            self.db_log_widget.insertHtml(formatted)
            self.db_log_widget.moveCursor(self.db_log_widget.textCursor().MoveOperation.End)

            if 'CRITICAL' in level or 'ALERT' in action or '긴급' in msg:
                self.trigger_alert_ui()

        except Exception as e:
            error_msg = f"<span style='color:red'>[LogUI Fatal Error] {e}</span><br>"
            self.db_log_widget.insertHtml(error_msg)
            print(f"[LogUI Error] {e}")
    
    def _blink_alert(self):
        """0.5초 간격으로 전체 배경만 붉게 점멸 (IMU 박스 정확히 제외)"""
        if not self.alert_active:
            return

        self.blink_state = not self.blink_state

        if self.blink_state:
            # 🔴 빨강 배경 표시
            self.alert_overlay.setStyleSheet("background-color: rgba(255, 0, 0, 255);")
            self.alert_overlay.show()

            # 🟦 IMU 박스 위치 정확히 제외하기
            imu_box = [g for g in self.findChildren(QGroupBox) if "IMU" in g.title()]
            if imu_box:
                # ① 전역 좌표로 변환
                imu_rect = imu_box[0].rect()
                imu_rect = imu_box[0].mapTo(self, imu_rect.topLeft())
                imu_abs_rect = QRect(imu_rect.x(), imu_rect.y(),
                                    imu_box[0].width(), imu_box[0].height())

                # ② 여백 살짝 확장 (덜 깎이게)
                padding = 8  # 너무 붙지 않게 여백 확보
                imu_abs_rect.adjust(-padding, -padding, padding, padding)

                # ③ 전체 창 기준으로 마스크 생성
                full_rect = self.rect()
                region = QRegion(full_rect)
                region -= QRegion(imu_abs_rect)
                self.alert_overlay.setMask(region)
            else:
                self.alert_overlay.clearMask()

            self.alert_overlay.lower()

        else:
            self.alert_overlay.setStyleSheet("background-color: transparent;")
            self.alert_overlay.clearMask()

    def trigger_alert_ui(self):
        """긴급 로그 감지 시 전체 GUI 깜빡임 시작"""
        if self.alert_active:
            return  # 이미 경보 중이면 무시

        self.alert_active = True
        print("[GUI] 🔴 CRITICAL ALERT 점멸 시작")

        # 0.5초 간격으로 깜빡임 시작
        self.alert_timer.start()

        # 3초 뒤 자동 복귀
        QTimer.singleShot(3000, self.reset_alert_ui)

    def reset_alert_ui(self):
        """3초 뒤 경보 해제"""
        if not self.alert_active:
            return

        self.alert_active = False
        self.alert_timer.stop()
        self.alert_overlay.setStyleSheet("background-color: transparent;")
        self.alert_overlay.hide()

        print("[GUI] ✅ ALERT 해제, 기본 상태로 복귀")

    def eventFilter(self, obj, event):
        if event.type() == event.Type.Resize:
            self.alert_overlay.setGeometry(0, 0, self.width(), self.height())
        return super().eventFilter(obj, event)


    def update_logbook_tab(self, data):
        """
        LOGBOOK 토픽 수신 시 항해일지 탭에 출력
        """
        try:
            entries = data.get("entries", [])
            if not entries:
                self.voyage_log_widget.setPlainText("최근 항해일지 데이터가 없습니다.")
                return

            text_lines = []
            for e in entries:
                text_lines.append(
                    f"[{e['log_dt']}] "
                    f"풍향: {e['wind_dir']} / 풍속: {e['wind_spd']} m/s / "
                    f"날씨: {e['weather']} / "
                    f"항로상태: {'ON' if e['on_route'] else 'OFF'}\n"
                    f"운항 메모: {e['on_notes']}\n"
                    f"특이사항: {e['ex_notes']}\n"
                    "-----------------------------------------"
                )

            self.voyage_log_widget.setPlainText("\n".join(text_lines))

        except Exception as e:
            print(f"[update_logbook_tab Error] {e}")
            self.voyage_log_widget.setPlainText(f"항해일지 데이터 표시 중 오류: {e}")
    
    # --- 카메라 업데이트 (QGraphicsView용) ---
    def update_camera_view(self, pixmap_item, base64_data):
        try:
            img_data = safe_b64decode(base64_data)
            qimg = QImage.fromData(img_data)
            if qimg.isNull():
                return

            pix = QPixmap.fromImage(qimg)
            pixmap_item.setPixmap(pix)

            # 🔹 장면 즉시 갱신
            scene = pixmap_item.scene()
            scene.update()

            # 🔹 화면 비율 맞춤 자동 스케일
            view = pixmap_item.scene().views()[0]
            view.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        except Exception as e:
            print(f"[Camera Feed Error] {e}")
# --- Entry Point ---
if __name__ == '__main__':
    if os.environ.get('XDG_RUNTIME_DIR') is None and 'root' in os.environ.get('HOME', ''):
        os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'

    app = QApplication(sys.argv)
    ex = MarineDashboardApp()
    ex.show()
    sys.exit(app.exec())
