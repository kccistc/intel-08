"""
라즈베리파이5 최적화 낙상 감지 + 상태 전이 분석 (MQTT Client)
MoveNet Thunder + Rule-based Fall Detection + State Transition + MQTT

*** 주요 수정 사항 (2025-10-17) ***
1. [CRITICAL FIX] MoveNetPose: 부정확한 크롭 및 좌표 변환 로직을 단순하고 정확한 방식으로 전면 교체.
2. [CRITICAL FIX] main-loop: YOLO 프레임 스킵 시 발생하던 트래커 업데이트 버그 수정.
3. [IMPROVEMENT] YOLO: imgsz를 416으로 조정하여 속도와 정확도 균형 개선.
4. [IMPROVEMENT] GUI: 키포인트 시각화 기능 추가하여 디버깅 편의성 증대.
"""

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timezone
import base64

# ============================================
# MQTT 설정
# ============================================
MQTT_BROKER = "10.10.14.73"  # 서버 IP로 변경하세요
MQTT_PORT = 1883
TOPIC_BASE = "project/vision"

# 🚨🚨 AD_USER 인증 정보 추가 🚨🚨
MQTT_USERNAME = "PE_USER"      # 등록된 AD 사용자 이름
MQTT_PASSWORD = "sksk"  # 등록된 AD 사용자 비밀번호 (실제 값으로 변경 필요)

# 모듈 이름 및 토픽 설정
FALL_MODULE = "PE"
RAW_TOPIC = TOPIC_BASE + "/" + FALL_MODULE + "/RAW"
ALERT_TOPIC = TOPIC_BASE + "/" + FALL_MODULE + "/ALERT"
FALL_VIDEO_TOPIC = "project/vision/FALL/VIDEO"  # 비디오 스트림 토픽

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ============================================
# 설정
# ============================================
DEBUG_MODE = True  # True로 하면 상세 로그

# 위험구역 설정
USE_RATIO = False
DANGER_X_MIN = None
DANGER_X_MAX = 200
DANGER_Y_MIN = None
DANGER_Y_MAX = None

DANGER_X_RATIO_MIN = None
DANGER_X_RATIO_MAX = 0.3
DANGER_Y_RATIO_MIN = None
DANGER_Y_RATIO_MAX = None

ZONE_WARNING_TIME = 3
ZONE_ALERT_TIME = 5
DANGER_AREA_COLOR = (0, 0, 255)

# 낙상 판단 설정
FALL_CONFIDENCE_THRESHOLD = 0.60
FALL_FRAMES = 3
FALL_TRANSITION_TIME = 1.0

# MQTT RAW 데이터 발행 주기 (프레임)
RAW_PUBLISH_INTERVAL = 15


# ============================================
# MoveNet 포즈 추정 (수정된 버전)
# ============================================
class MoveNetPose:
    """라즈베리파이5용 MoveNet Thunder (정확한 좌표 복원 포함)"""

    def __init__(self, model_type='thunder', device='cpu'):
        print(f"Loading MoveNet {model_type}...")
        self.model_type = model_type
        self.input_size = 256 if model_type == 'thunder' else 192
        model_path = f'movenet_{model_type}.tflite'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MoveNet TFLite model not found: {model_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        print(f"[{now_str()}] ✅ MoveNet loaded: {model_path}")

    def predict(self, frame, bboxes, scores=None):
        if not hasattr(self, 'interpreter') or bboxes is None or len(bboxes) == 0:
            return []

        poses = []
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        model_dtype = input_details[0]['dtype']

        frame_h, frame_w, _ = frame.shape

        for bbox in bboxes:
            x1_orig, y1_orig, x2_orig, y2_orig = bbox.astype(int)

            # --- [수정] ✅ 정확하고 효율적인 정사각형 크롭 로직 ---
            # 1. 바운딩 박스의 중심과 크기 계산
            center_x = (x1_orig + x2_orig) / 2
            center_y = (y1_orig + y2_orig) / 2
            bbox_w = x2_orig - x1_orig
            bbox_h = y2_orig - y1_orig

            # 2. 정사각형을 만들기 위한 최대 길이 선택
            square_size = max(bbox_w, bbox_h)

            # 3. 정사각형 크롭 좌표 계산 (프레임 경계 확인)
            x1_crop = max(0, int(center_x - square_size / 2))
            y1_crop = max(0, int(center_y - square_size / 2))
            x2_crop = min(frame_w, int(center_x + square_size / 2))
            y2_crop = min(frame_h, int(center_y + square_size / 2))

            # 4. 이미지 크롭
            crop_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]
            crop_h, crop_w, _ = crop_img.shape
            if crop_h == 0 or crop_w == 0:
                continue

            # --- [수정] ✅ 모델 입력 전처리 및 타입 핸들링 ---
            input_tensor = tf.image.resize(crop_img, [self.input_size, self.input_size])

            if model_dtype == np.float32:
                input_tensor = tf.cast(input_tensor, tf.float32) / 255.0
            else: # uint8, int8
                input_tensor = tf.cast(input_tensor, model_dtype)

            input_tensor = tf.expand_dims(input_tensor, axis=0)

            # --- Inference ---
            self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            keypoints_with_scores = np.squeeze(self.interpreter.get_tensor(output_details[0]['index']))

            # --- [수정] ✅ 단순하고 정확한 좌표 역변환 ---
            # 모델 출력(0~1) -> 크롭 이미지 좌표 -> 원본 프레임 좌표
            kp_y = keypoints_with_scores[:, 0] * crop_h + y1_crop
            kp_x = keypoints_with_scores[:, 1] * crop_w + x1_crop
            kp_score = keypoints_with_scores[:, 2]

            keypoints = np.stack([kp_x, kp_y, kp_score], axis=1)

            proposal_score = float(np.mean(keypoints[:, 2]))
            poses.append({'keypoints': keypoints, 'proposal_score': proposal_score, 'bbox': bbox})

        return poses


# ============================================
# Rule 기반 자세 분류
# ============================================

def estimate_motion(prev_kp, curr_kp):
    """평균 키포인트 이동량"""
    if prev_kp is None or len(prev_kp) == 0:
        return 0.0

    valid = (prev_kp[:, 2] > 0.2) & (curr_kp[:, 2] > 0.2)
    if np.sum(valid) < 5:
        return 0.0

    diffs = np.linalg.norm(curr_kp[valid, :2] - prev_kp[valid, :2], axis=1)
    motion = float(np.mean(diffs))

    return motion


# ============================================
# Rule 기반 자세 분류 (수정된 버전)
# ============================================
def detect_fall_rule_based(keypoints, prev_keypoints=None):
    """몸통 각도를 핵심으로 사용하는 안정적인 자세 분류 로직"""

    conf = float(np.mean(keypoints[:, 2]))
    valid_kp = keypoints[keypoints[:, 2] > 0.2]

    if len(valid_kp) < 5:
        return 'Unknown', conf * 0.3, {}

    # 1. 기본 특징
    width = valid_kp[:, 0].max() - valid_kp[:, 0].min()
    height = valid_kp[:, 1].max() - valid_kp[:, 1].min()
    ratio = width / (height + 1e-6)

    # 2. 주요 관절 위치
    shoulder_y = [keypoints[i][1] for i in [5, 6] if keypoints[i][2] > 0.2]
    shoulder_x = [keypoints[i][0] for i in [5, 6] if keypoints[i][2] > 0.2]
    hip_y = [keypoints[i][1] for i in [11, 12] if keypoints[i][2] > 0.15]
    hip_x = [keypoints[i][0] for i in [11, 12] if keypoints[i][2] > 0.15]
    ankle_y = [keypoints[i][1] for i in [15, 16] if keypoints[i][2] > 0.15]
    
    # 3. 핵심 특징 계산
    ha_dist = abs(np.mean(ankle_y) - np.mean(hip_y)) if len(hip_y) > 0 and len(ankle_y) > 0 else 0
    lower_ratio = ha_dist / (height + 1e-6)
    
    torso_angle = 0
    torso_vertical = False
    if len(shoulder_y) > 0 and len(hip_y) > 0 and len(shoulder_x) > 0 and len(hip_x) > 0:
        torso_height = abs(np.mean(shoulder_y) - np.mean(hip_y))
        torso_width = abs(np.mean(shoulder_x) - np.mean(hip_x))
        torso_angle = np.degrees(np.arctan2(torso_width, torso_height + 1e-6))
        # 몸통이 수직에 가까운가? (각도가 35도 미만)
        if torso_angle < 35:
            torso_vertical = True

    motion = estimate_motion(prev_keypoints, keypoints) if prev_keypoints is not None else 0

    details = {
        "ratio": f"{ratio:.2f}", "height": f"{height:.0f}", "lower_r": f"{lower_ratio:.2f}",
        "torso_angle": f"{torso_angle:.1f}", "motion": f"{motion:.1f}"
    }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔴 1순위: Lying Down (누움/낙상) - 몸통 각도가 핵심 증거
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 조건: 몸통이 거의 수평(각도 60도 이상)이고, 전체 비율도 넓어야 함
    if torso_angle > 60 and ratio > 1.0:
        return 'Lying Down', conf * 0.98, details
    # 조건: 몸통이 꽤 기울었고(45도 이상), 높이가 매우 낮음
    if torso_angle > 45 and height < 120 and ratio > 0.9:
        return 'Lying Down', conf * 0.95, details

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🟡 2순위: Sitting (앉음) - 수직 몸통이 핵심 증거
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 조건: 몸통이 수직이고, 하체 비율이 낮음 (다리가 접혀있음)
    if torso_vertical and lower_ratio < 0.38:
        return 'Sitting', conf * 0.95, details
    # 조건: 몸통이 수직이고, 전체 키가 작음
    if torso_vertical and height < 150 and ratio > 0.5:
        return 'Sitting', conf * 0.90, details

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🟢 3순위: Standing / Walking (서있음/걸음)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 조건: 몸통이 수직이고, 하체 비율이 높음 (다리가 펴져있음)
    if torso_vertical and lower_ratio >= 0.38:
        if motion > 6:
            return 'Walking', conf * 0.95, details
        else:
            return 'Standing', conf * 0.95, details
    
    # 위 모든 명확한 규칙에 해당하지 않으면 'Unknown'
    return 'Unknown', conf * 0.5, details


# ============================================
# 위험구역 함수
# ============================================

def get_location_details(bbox):
    """바운딩 박스를 이용해 중심 x, 하단 y 좌표를 반환"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    bottom_y = y2
    return int(center_x), int(bottom_y)


def is_in_danger_zone(bbox, frame_width, frame_height):
    """바운딩박스가 위험구역에 있는지 확인"""
    center_x, bottom_y = get_location_details(bbox)

    if USE_RATIO:
        x_min = int(frame_width * DANGER_X_RATIO_MIN) if DANGER_X_RATIO_MIN is not None else 0
        x_max = int(frame_width * DANGER_X_RATIO_MAX) if DANGER_X_RATIO_MAX is not None else frame_width
        y_min = int(frame_height * DANGER_Y_RATIO_MIN) if DANGER_Y_RATIO_MIN is not None else 0
        y_max = int(frame_height * DANGER_Y_RATIO_MAX) if DANGER_Y_RATIO_MAX is not None else frame_height
    else:
        x_min = DANGER_X_MIN if DANGER_X_MIN is not None else 0
        x_max = DANGER_X_MAX if DANGER_X_MAX is not None else frame_width
        y_min = DANGER_Y_MIN if DANGER_Y_MIN is not None else 0
        y_max = DANGER_Y_MAX if DANGER_Y_MAX is not None else frame_height

    in_danger_x = True
    if x_min is not None and center_x < x_min:
        in_danger_x = False
    if x_max is not None and center_x > x_max:
        in_danger_x = False

    in_danger_y = True
    if y_min is not None and bottom_y < y_min:
        in_danger_y = False
    if y_max is not None and bottom_y > y_max:
        in_danger_y = False

    return in_danger_x and in_danger_y


def draw_danger_area(frame, show_gui):
    """위험구역 시각화 (GUI 모드일 때만)"""
    if not show_gui:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    if USE_RATIO:
        x_min = int(w * DANGER_X_RATIO_MIN) if DANGER_X_RATIO_MIN is not None else 0
        x_max = int(w * DANGER_X_RATIO_MAX) if DANGER_X_RATIO_MAX is not None else w
        y_min = int(h * DANGER_Y_RATIO_MIN) if DANGER_Y_RATIO_MIN is not None else 0
        y_max = int(h * DANGER_Y_RATIO_MAX) if DANGER_Y_RATIO_MAX is not None else h
    else:
        x_min = DANGER_X_MIN if DANGER_X_MIN is not None else 0
        x_max = DANGER_X_MAX if DANGER_X_MAX is not None else w
        y_min = DANGER_Y_MIN if DANGER_Y_MIN is not None else 0
        y_max = DANGER_Y_MAX if DANGER_Y_MAX is not None else h

    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), DANGER_AREA_COLOR, -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), DANGER_AREA_COLOR, 3)
    cv2.putText(frame, "DANGER ZONE", (x_min + 10, y_min + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, DANGER_AREA_COLOR, 2)

    return frame


def draw_zone_warnings(frame, zone_warnings, show_gui):
    """여러 위험구역 경고를 순차적으로 표시 (GUI 모드일 때만)"""
    if not show_gui:
        return frame

    y_offset = 50

    for i, (track_id, elapsed_time, status) in enumerate(zone_warnings):
        if status == 'warning':
            color = (0, 255, 255)
            text = f"WARNING! Worker #{track_id} in danger zone {elapsed_time:.1f}s"
        elif status == 'danger':
            color = (0, 0, 255)
            text = f"DANGER! Worker #{track_id} in danger zone {elapsed_time:.1f}s"
        else:
            continue

        y_pos = y_offset + (i * 35)
        cv2.rectangle(frame, (10, y_pos), (650, y_pos + 30), color, -1)
        cv2.putText(frame, text, (15, y_pos + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

# ============================================
# Improved Tracker
# ============================================
class ImprovedTracker:
    """IoU 기반 트래커 (필터 완화 버전)"""

    def __init__(self, max_age=12):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.frame_count = 0

    def _is_valid_person(self, keypoints, bbox):
        avg_conf = np.mean(keypoints[:, 2])
        if avg_conf < 0.15:
            return False
        important = [5, 6, 11, 12]
        imp_conf = [keypoints[i][2] for i in important if i < len(keypoints)]
        if len(imp_conf) < 2 or np.mean(imp_conf) < 0.25:
            return False
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w < 20 or h < 30 or h / (w + 1e-9) < 0.15:
            return False
        return True

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 < x1 or y2 < y1: return 0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (area1 + area2 - inter + 1e-9)

    def update(self, detections):
        self.frame_count += 1
        for tid in list(self.tracks.keys()):
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        valid = [d for d in detections if self._is_valid_person(d['keypoints'], d['bbox'])]
        for det in valid:
            best_iou, best_id = 0, None
            for tid, tr in self.tracks.items():
                iou = self._iou(det['bbox'], tr['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou, best_id = iou, tid
            if best_id:
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['keypoints'].append(det['keypoints'])
                self.tracks[best_id]['age'] = 0
            else:
                self.tracks[self.next_id] = {'bbox': det['bbox'], 'keypoints': [det['keypoints']], 'age': 0}
                self.next_id += 1
        return list(self.tracks.keys())

# ============================================
# YOLOv8 검출기 (안정화 버전)
# ============================================
class ImprovedYOLODetector:
    """YOLOv8n 사람 검출 - 호환성 및 안정성 개선"""

    def __init__(self, model_name='yolov8n.pt', conf_thres=0.45, device='cpu'): # conf_thres 살짝 낮춤
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.device = device

    def detect(self, frame):
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = self.model.predict(
            source=frame, conf=self.conf_thres, imgsz=416,
            classes=[0], device=self.device, verbose=False
        )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.array([]), np.array([])

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        
        bboxes, scores = [], []
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            w, h = x2 - x1, y2 - y1
            
            # [수정] ✅ 필터링 조건 대폭 완화
            # 최소 크기 조건만 남김
            if w < 20 or h < 20:
                continue
            
            bboxes.append([x1, y1, x2, y2])
            scores.append(conf)

        if DEBUG_MODE:
            # YOLO 탐지 결과가 10프레임에 한번씩만 보이도록 수정 (터미널 깔끔하게)
            if hasattr(self, 'frame_counter'):
                self.frame_counter += 1
            else:
                self.frame_counter = 1
            
            if self.frame_counter % 10 == 0:
                print(f"[YOLO] Detected {len(bboxes)} persons")

        return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)


# ============================================
# MQTT 발행 함수
# ============================================
def on_connect(client, userdata, flags, rc):
    """MQTT 연결 콜백"""
    if rc == 0:
        print(f"[{now_str()}] ✅ MQTT Connected successfully.")
    else:
        print(f"[{now_str()}] ❌ MQTT Connection failed with code {rc}")

def publish_mqtt_message(client, topic, payload):
    """JSON 메시지를 MQTT로 발행"""
    try:
        json_payload = json.dumps(payload, ensure_ascii=False)
        client.publish(topic, json_payload, qos=0)
        if DEBUG_MODE:
            msg_type = "ALERT" if "ALERT" in topic else "RAW"
            level = payload.get("level", "N/A")
            print(f"[{now_str()}] [MQTT SEND - {msg_type}:{level}] {topic}")
    except Exception as e:
        print(f"[{now_str()}] [MQTT ERROR] Failed to publish to {topic}: {e}")


# ============================================
# 메인 함수
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Improved Fall Detection with MQTT Client')
    parser.add_argument('--camera', type=str, default='0', help='Camera source')
    parser.add_argument('--device', type=str, default='cpu', help='cpu only for RPi5')
    parser.add_argument('--model', type=str, default='thunder', choices=['thunder', 'lightning'])
    parser.add_argument('--show-gui', action='store_true', help='Show GUI window with visualizations')
    args = parser.parse_args()

    print("="*60)
    print("Improved Fall Detection System (MQTT Client)")
    print("- Object Detection: YOLOv8n (Enhanced Filtering)")
    print("- Pose: MoveNet " + args.model.title())
    print("- Device: CPU")
    print(f"- MQTT Broker: {MQTT_BROKER}:{MQTT_PORT} / Module: {FALL_MODULE}")
    print(f"- GUI Mode: {'Enabled' if args.show_gui else 'Disabled'}")
    print("="*60)

    # 모델 로드
    print("\n1️⃣ Loading models...")
    detector = ImprovedYOLODetector(model_name='yolov8n.pt', conf_thres=0.5, device='cpu')
    pose_model = MoveNetPose(model_type=args.model)

    # 트래커
    tracker = ImprovedTracker(max_age=12)

    # MQTT 클라이언트 초기화
    mqtt_client = mqtt.Client(client_id="PE_Client", protocol=mqtt.MQTTv311)

    # 사용자 이름 및 비밀번호 설정
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[{now_str()}] ❌ Failed to connect to MQTT broker: {e}")

    # 카메라
    print("\n2️⃣ Opening camera...")
    cam_source = args.camera
    if cam_source.isdigit():
        cap = cv2.VideoCapture(int(cam_source))
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"[{now_str()}] ❌ Cannot open camera!")
        mqtt_client.loop_stop()
        return

    print(f"[{now_str()}] ✅ Camera opened")

    # 상태 변수
    fall_counters = {}
    fall_alerted = {}
    zone_timers = {}
    previous_states = {}
    alert_sent_zone = {}

    fps_time = time.time()
    frame_count = 0

    # [수정] ✅ YOLO 프레임 스킵 및 결과 저장을 위한 변수
    yolo_skip_counter = 0
    YOLO_SKIP_FRAMES = 3
    last_tracked_data = {} # 추적 결과를 저장할 변수

    quit_msg = "Press 'q' to quit" if args.show_gui else "Press Ctrl+C to quit"
    print(f"[{now_str()}] 3️⃣ Starting detection... ({quit_msg})\n")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"[{now_str()}] End of stream or video.")
                break

            frame_count += 1
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # 위험구역 그리기 (매 프레임)
            frame = draw_danger_area(frame, args.show_gui)

            # [수정] ✅ 프레임 스킵 로직
            yolo_skip_counter += 1
            if yolo_skip_counter >= YOLO_SKIP_FRAMES:
                yolo_skip_counter = 0 # 카운터 리셋
                last_tracked_data = {} # 이전 추적 데이터 초기화

                # --- 이 블록 안에서만 탐지/추적 파이프라인 전체를 실행 ---
                bboxes, scores = detector.detect(frame)
                
                detections = []
                if len(bboxes) > 0:
                    poses = pose_model.predict(frame, bboxes, scores)
                    for pose, bbox in zip(poses, bboxes):
                        detections.append({
                            'bbox': bbox,
                            'keypoints': pose['keypoints'],
                            'score': pose['proposal_score']
                        })
                
                current_tracks_ids = tracker.update(detections)
                
                # --- 추적된 객체들의 최종 정보를 저장 ---
                for track_id in current_tracks_ids:
                    if track_id in tracker.tracks:
                        track = tracker.tracks[track_id]
                        if len(track['keypoints']) > 0:
                            last_tracked_data[track_id] = {
                                'bbox': track['bbox'],
                                'keypoints_list': track['keypoints'] # 상태 분석을 위해 리스트 전체 저장
                            }

            # --- [수정] ✅ 매 프레임 상태 분석 및 시각화는 저장된 'last_tracked_data' 사용 ---
            current_tracks = list(last_tracked_data.keys())
            raw_detections_list = []
            is_person_detected = len(current_tracks) > 0

            # 각 트랙 처리
            for track_id in current_tracks:
                tracked_info = last_tracked_data[track_id]
                bbox = tracked_info['bbox']
                keypoints_list = tracked_info['keypoints_list']
                
                if len(keypoints_list) < 1:
                    continue

                current_kp = keypoints_list[-1]
                prev_kp = keypoints_list[-2] if len(keypoints_list) > 1 else None
                
                x1, y1, x2, y2 = bbox.astype(int)
                center_x, bottom_y = get_location_details(bbox)

                # --- [A] 위험구역 체크 (ZONE) ---
                in_zone = is_in_danger_zone(bbox, w, h)
                
                if in_zone:
                    if track_id not in zone_timers:
                        zone_timers[track_id] = current_time
                        alert_sent_zone[track_id] = False
                    
                    elapsed = current_time - zone_timers[track_id]
                    
                    if elapsed >= ZONE_ALERT_TIME and not alert_sent_zone.get(track_id, False):
                        alert_payload = {
                            "timestamp": now_str(), "module": FALL_MODULE, "level": "CRITICAL",
                            "message": f"🚨 DANGER ZONE CRITICAL: Worker #{track_id} in high-risk area for {elapsed:.1f}s.",
                            "details": [{"track_id": track_id, "action": "InDangerZoneCritical", "location": f"({center_x}, {bottom_y})"}]
                        }
                        publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                        alert_sent_zone[track_id] = True
                else:
                    if track_id in zone_timers:
                        del zone_timers[track_id]
                        alert_sent_zone[track_id] = False
                
                # --- [B] 낙상 감지 (FALL) ---
                rule_action, rule_conf, details = detect_fall_rule_based(current_kp, prev_kp)
                
                # 🔴 상태 전이 분석
                prev_action = previous_states.get(track_id, {}).get('action', None)
                state_start_time = previous_states.get(track_id, {}).get('state_start_time', current_time)
                
                final_action = rule_action
                new_state_start_time = state_start_time

                if rule_action == 'Lying Down':
                    final_action = 'Fall Down'
                    if prev_action != 'Fall Down':
                        new_state_start_time = current_time
                elif prev_action == 'Fall Down' and rule_action in ['Standing', 'Walking', 'Sitting']:
                    new_state_start_time = current_time
                elif prev_action != rule_action:
                    new_state_start_time = current_time
                
                previous_states[track_id] = {
                    'action': final_action,
                    'state_start_time': new_state_start_time
                }
                
                # 낙상 카운터
                if final_action == 'Fall Down' and rule_conf >= FALL_CONFIDENCE_THRESHOLD:
                    fall_counters[track_id] = fall_counters.get(track_id, 0) + 1
                else:
                    fall_counters[track_id] = 0
                    fall_alerted[track_id] = False

                # 🚨 FALL CRITICAL ALERT (낙상 확정)
                if fall_counters.get(track_id, 0) >= FALL_FRAMES and not fall_alerted.get(track_id, False):
                    alert_payload = {
                        "timestamp": now_str(), "module": FALL_MODULE, "level": "CRITICAL",
                        "message": f"🚨 CRITICAL FALL DETECTED: Worker #{track_id} is {final_action.lower()}.",
                        "details": [{"track_id": track_id, "action": final_action, "confidence": float(rule_conf), "location": f"({center_x}, {bottom_y})"}]
                    }
                    publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                    fall_alerted[track_id] = True
                    print(f"[{now_str()}] 🚨🚨 [CRITICAL ALERT SENT] Fall alert for Worker #{track_id}")

                # RAW 데이터 기록
                raw_detections_list.append({
                    "track_id": int(track_id), "object_type": "Person", "action": final_action,
                    "confidence": float(rule_conf), "x_center": center_x, "y_bottom": bottom_y,
                    "in_danger_zone": in_zone
                })

            # GUI 모드: 화면에 시각화
            if args.show_gui:
                for track_id in current_tracks:
                    tracked_info = last_tracked_data[track_id]
                    bbox = tracked_info['bbox']
                    current_kp = tracked_info['keypoints_list'][-1]
                    
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    final_action = previous_states.get(track_id, {}).get('action', 'Unknown')
                    
                    if fall_counters.get(track_id, 0) >= FALL_FRAMES: clr = (0, 0, 255)
                    elif final_action == 'Fall Down': clr = (0, 0, 255)
                    elif final_action == 'Standing': clr = (0, 255, 0)
                    elif final_action == 'Sitting': clr = (0, 255, 255)
                    elif final_action == 'Walking': clr = (255, 255, 0)
                    else: clr = (255, 255, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
                    cv2.putText(frame, f'ID:{track_id} {final_action}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
                    
                    # [추가] ✅ 키포인트 시각화
                    for i in range(len(current_kp)):
                        if current_kp[i, 2] > 0.3:
                            kx, ky = int(current_kp[i, 0]), int(current_kp[i, 1])
                            cv2.circle(frame, (kx, ky), 3, (255, 0, 255), -1)

                fps = 1.0 / (time.time() - fps_time + 1e-6)
                info_text = f'FPS: {fps:.1f} | Persons: {len(current_tracks)}'
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Fall Detection System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 5. 비디오 스트림 전송
            try:
                ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if ret_enc:
                    jpg_as_text = base64.b64encode(buffer.tobytes())
                    mqtt_client.publish(FALL_VIDEO_TOPIC, jpg_as_text, qos=0)
            except Exception as e:
                print(f"[{now_str()}] [ERROR] ❌ FALL Video streaming publish failed: {e}")

            # 6. RAW 데이터 발행 (주기적으로)
            if frame_count % RAW_PUBLISH_INTERVAL == 0 and is_person_detected:
                raw_payload = {
                    "timestamp": now_str(),
                    "module": FALL_MODULE,
                    "level": "INFO",
                    "detections": raw_detections_list,
                    "person_detected": is_person_detected
                }
                mqtt_client.publish(RAW_TOPIC, json.dumps(raw_payload, ensure_ascii=False), qos=0)
            
            fps_time = time.time()
            time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"\n[{now_str()}] [INFO System] Program stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"\n[{now_str()}] [ERROR System] An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("\n" + "="*60 + "\nProgram terminated.\n" + "="*60)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()