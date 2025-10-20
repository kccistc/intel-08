import os
import cv2
import time
import torch
import argparse
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json
from datetime import datetime, timezone
import base64

# ====================================================
# 0. 고정 카메라 할당을 위한 모듈 임포트 (수정된 부분 1/2)
# camera_init_robust.py 파일이 이 스크립트와 같은 경로에 있어야 합니다.
# ====================================================
from camera_init_robust import find_camera_by_vid_pid 

# ============================================
# MQTT 설정 (서버와 동일해야 합니다)
# ============================================
MQTT_BROKER = "10.10.14.73" 
MQTT_PORT = 1883
TOPIC_BASE = "project/vision"

# 🚨🚨 PE_USER 인증 정보 추가 🚨🚨
MQTT_USERNAME = "PE_USER"      # 등록된 PE 사용자 이름
MQTT_PASSWORD = "sksk"  # 등록된 PE 사용자 비밀번호

# 수정: 모듈 이름 및 토픽 분리
PE_MODULE = "PE"
RAW_TOPIC = TOPIC_BASE + "/" + PE_MODULE + "/RAW"
ALERT_TOPIC = TOPIC_BASE + "/" + PE_MODULE + "/ALERT" # 경고 토픽도 AD 전용으로 분리
PE_VIDEO_TOPIC = "project/vision/PE/VIDEO" # 시연용 비디오 스트림 토픽

def now_str():
    """ISO 8601 형식의 현재 UTC 시각을 반환합니다."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ============================================
# 설정
# ============================================
# 디버그 모드는 True로 유지하여 로그 출력은 계속합니다.
DEBUG_MODE = True  # True로 하면 상세 로그

# 위험구역 설정 (절대 좌표)
USE_RATIO = False
DANGER_X_MIN = None
DANGER_X_MAX = 200
DANGER_Y_MIN = None
DANGER_Y_MAX = None

# 비율 방식 (USE_RATIO = True일 때만 사용)
DANGER_X_RATIO_MIN = None
DANGER_X_RATIO_MAX = 0.3
DANGER_Y_RATIO_MIN = None
DANGER_Y_RATIO_MAX = None

ZONE_WARNING_TIME = 3
ZONE_ALERT_TIME = 5
SHOW_DANGER_AREA = False # 화면 출력을 제거했으므로 이 플래그는 사용하지 않지만 로직은 유지합니다.
DANGER_AREA_COLOR = (0, 0, 255)

# 넘어짐 판단 설정
FALL_CONFIDENCE_THRESHOLD = 0.65
FALL_FRAMES = 3 # 이 프레임 수만큼 연속되어야 낙상 확정

# MQTT RAW 데이터 발행 주기 (프레임)
RAW_PUBLISH_INTERVAL = 15

# ============================================
# MoveNet 포즈 추정 모델
# ============================================
class MoveNetPose:
    """MoveNet Thunder - 라즈베리파이5 최적화"""
    
    def __init__(self, model_type='thunder', device='cpu'):
        print(f"Loading MoveNet {model_type}...")
        self.model_type = model_type
        self.input_size = 256 if model_type == 'thunder' else 192
        self.use_tflite = False
        
        # TFLite 파일을 우선 시도
        model_path = f'movenet_{model_type}.tflite'
        if os.path.exists(model_path):
            try:
                # TFLite 모델 로드
                # WARNING: TensorFlow Lite Interpreter는 RPi에서 실행 시 CPU 최적화가 중요합니다.
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.use_tflite = True
                print(f"[{now_str()}] ✅ Loaded TFLite model successfully!")
                return
            except Exception as e:
                print(f"[{now_str()}] ❌ TFLite loading failed: {e}")
                
        # TFLite 로드 실패 시, 최소한의 기능은 유지 (TFLite 로드가 성공해야 작동)
        if not self.use_tflite:
             print(f"[{now_str()}] ⚠️ MoveNet TFLite model not found or failed to load. Pose estimation disabled.")

    
    def predict(self, frame, bboxes, scores=None):
        """
        프레임 안의 여러 사람에 대한 자세를 예측합니다.
        """
        if bboxes is None or len(bboxes) == 0:
            return []

        poses = []
        # TFLite Interpreter가 로드되었는지 확인
        if not hasattr(self, 'interpreter') or not self.use_tflite:
            return []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = frame[y1:y2, x1:x2]
            crop_height, crop_width, _ = crop_img.shape
            
            # 텐서플로우 변환 (TFLite 사용 시 필요)
            input_image = tf.convert_to_tensor(crop_img, dtype=tf.uint8)
            
            # 모델 입력에 맞게 리사이즈 및 패딩
            resized_img = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
            input_img_tensor = tf.cast(resized_img, dtype=tf.uint8)
            input_batch = tf.expand_dims(input_img_tensor, axis=0)
            
            # TFLite Inference
            input_details = self.interpreter.get_input_details()
            self.interpreter.set_tensor(input_details[0]['index'], input_batch)
            self.interpreter.invoke()
            
            output_details = self.interpreter.get_output_details()
            keypoints_with_scores = np.squeeze(self.interpreter.get_tensor(output_details[0]['index']))
            
            # 좌표 변환 로직 (Movenet의 출력 좌표를 원본 이미지 좌표로 변환)
            if crop_height > crop_width:
                scale_factor = self.input_size / crop_height
                new_width = crop_width * scale_factor
                padd_x = (self.input_size - new_width) / 2
                padd_y = 0
                scale_x = new_width / crop_width
                scale_y = self.input_size / crop_height
            else:
                scale_factor = self.input_size / crop_width
                new_height = crop_height * scale_factor
                padd_x = 0
                padd_y = (self.input_size - new_height) / 2
                scale_x = self.input_size / crop_width
                scale_y = new_height / crop_height

            keypoints = np.zeros((17, 3), dtype=np.float32)
            
            norm_y = keypoints_with_scores[:, 0]
            norm_x = keypoints_with_scores[:, 1]
            
            # 변환된 좌표를 원본 bbox 위치에 맞게 조정
            keypoints[:, 0] = ((norm_x * self.input_size - padd_x) / scale_x) + x1
            keypoints[:, 1] = ((norm_y * self.input_size - padd_y) / scale_y) + y1
            keypoints[:, 2] = keypoints_with_scores[:, 2] # Confidence score
            
            proposal_score = float(np.mean(keypoints[:, 2]))
            
            poses.append({
                'keypoints': keypoints,  # [17, 3] 형태로 통일
                'proposal_score': proposal_score,
                'bbox': bbox
            })
        return poses

# ============================================
# 4. 룰 기반 낙상 감지 및 유틸리티 함수
# ============================================
def estimate_motion(prev_kp, curr_kp):
    """평균 키포인트 이동량 (걷기 인식용)"""
    if prev_kp is None or len(prev_kp) == 0 or prev_kp.shape != curr_kp.shape:
        return 0.0
    
    # 신뢰도 체크
    valid = (prev_kp[:, 2] > 0.2) & (curr_kp[:, 2] > 0.2)
    if np.sum(valid) < 5:
        return 0.0
    
    # 유효한 키포인트만 사용하여 이동량 계산
    diffs = np.linalg.norm(curr_kp[valid, :2] - prev_kp[valid, :2], axis=1)
    motion = float(np.mean(diffs))
    
    if DEBUG_MODE:
        print(f"    Motion calculation: {np.sum(valid)} valid points, motion={motion:.2f}")
    
    return motion

def calculate_body_angle(keypoints):
    """몸의 기울기 각도 (0° = 완전 수직, 90° = 완전 수평)"""
    if len(keypoints) < 13:
        return None

    valid_shoulder = []
    valid_hip = []

    # 어깨 중심점 계산
    if keypoints[5][2] > 0.2: # Left Shoulder
        valid_shoulder.append(keypoints[5][:2])
    if keypoints[6][2] > 0.2: # Right Shoulder
        valid_shoulder.append(keypoints[6][:2])
        
    # 골반 중심점 계산
    if keypoints[11][2] > 0.2: # Left Hip
        valid_hip.append(keypoints[11][:2])
    if keypoints[12][2] > 0.2: # Right Hip
        valid_hip.append(keypoints[12][:2])

    if len(valid_shoulder) == 0 or len(valid_hip) == 0:
        return None

    shoulder_center = np.mean(valid_shoulder, axis=0)
    hip_center = np.mean(valid_hip, axis=0)

    dx = hip_center[0] - shoulder_center[0]
    dy = shoulder_center[1] - hip_center[1] # y축은 아래로 갈수록 커지므로 어깨 y - 골반 y로 계산
    
    # 수직(y축)과의 각도 계산 (0도: 수직, 90도: 수평)
    angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
    return angle


def get_body_aspect_ratio(keypoints):
    """몸의 가로/세로 비율"""
    valid_points = keypoints[keypoints[:, 2] > 0.15]
    
    if len(valid_points) < 3:
        return None
    
    x_coords = valid_points[:, 0]
    y_coords = valid_points[:, 1]
    
    width = x_coords.max() - x_coords.min()
    height = y_coords.max() - y_coords.min()
    
    if height < 5:
        return None
    
    return width / height


def detect_fall_rule_based(keypoints, prev_keypoints=None):
    """향상된 룰 기반 상태 인식"""
    
    angle = calculate_body_angle(keypoints)
    ratio = get_body_aspect_ratio(keypoints)
    motion = estimate_motion(prev_keypoints, keypoints)

    conf = float(np.mean(keypoints[:, 2]))

    if angle is None or ratio is None:
        return 'Unknown', conf, {}
    
    details = {"angle": f"{angle:.1f}", "ratio": f"{ratio:.2f}", "motion": f"{motion:.1f}"}
    
    if DEBUG_MODE:
        print(f"  DEBUG >>> Angle: {angle:.1f}°, Ratio: {ratio:.2f}, Motion: {motion:.1f}")

    # 1. 눕거나 넘어진 상태 (수평에 가까움, angle > 50)
    if angle > 50:
        if ratio > 1.0:
            if motion < 7:
                return 'Lying Down', conf * 0.95, details
            else:
                # 높은 이동량 + 수평 자세 = 넘어지는 중 (Fall Down)
                return 'Fall Down', conf * 1.0, details
        else:
            return 'Unknown', conf, details

    # 2. 앉은 상태 (어느 정도 기울어짐, 10 <= angle < 50)
    elif 10 <= angle < 50 and ratio > 0.5:
        return 'Sitting', conf * 0.9, details
    
    # 3. 서 있거나 걷는 상태 (거의 수직, angle < 10)
    elif angle < 10:
        if motion > 2:
            return 'Walking', conf, details
        else:
            return 'Standing', conf, details
    
    else: 
        return 'Unknown', conf, details

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

# 시각화 더미 함수
def draw_danger_area(frame):
    return frame

def draw_zone_warnings(frame, zone_warnings):
    return frame

# ============================================
# 간단한 트래커 (IoU 기반)
# ============================================

class SimpleTracker:
    """간단한 IoU 기반 트래커"""
    
    def __init__(self, max_age=50):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
    
    def update(self, detections):
        """
        detections: List of {'bbox': [x1,y1,x2,y2], 'keypoints': array, 'score': float}
        """
        # 기존 트랙 나이 증가
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # IoU 매칭
        for det in detections:
            best_iou = 0
            best_id = None
            
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(det['bbox'], track['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                # 기존 트랙 업데이트
                self.tracks[best_id]['bbox'] = det['bbox']
                self.tracks[best_id]['keypoints'].append(det['keypoints'])
                if len(self.tracks[best_id]['keypoints']) > 30:
                    self.tracks[best_id]['keypoints'].pop(0)
                self.tracks[best_id]['age'] = 0
            else:
                # 새 트랙 생성
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'keypoints': [det['keypoints']],
                    'age': 0
                }
                self.next_id += 1
        
        # 현재 활성 트랙 ID 리스트 반환
        return list(self.tracks.keys())
    
    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

# ============================================
# YOLOv8 검출기
# ============================================

class YOLOv8_Detector:
    def __init__(self, model_name='yolov8n.pt', conf_thres=0.65, device='cpu'):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.device = device
    
    def detect(self, frame):
        """사람 검출"""
        results = self.model.predict(
            frame,
            conf=self.conf_thres,
            classes=[0],  # person only
            device=self.device,
            verbose=False
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return [], []
        
        boxes = results[0].boxes
        bboxes = []
        scores = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            bboxes.append([x1, y1, x2, y2])
            scores.append(conf)
        
        return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)

# 시각화 더미 함수
def draw_skeleton(frame, keypoints):
    return frame

# ============================================
# MQTT 발행 함수
# (기존 코드와 동일)
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
            # 수정: RAW/ALERT에 따라 다른 로그 메시지 출력
            msg_type = "ALERT" if "ALERT" in topic else "RAW"
            level = payload.get("level", "N/A")
            print(f"[{now_str()}] [MQTT SEND - {msg_type}:{level}] {topic}")
    except Exception as e:
        print(f"[{now_str()}] [MQTT ERROR] Failed to publish to {topic}: {e}")


# ============================================
# 메인
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 Optimized Fall Detection')
    # --- [!] 카메라 인덱스를 args.camera 대신 find_camera_by_vid_pid로 결정 ---
    parser.add_argument('--camera', type=str, default='0', help='Camera source or video path (This is now overridden by fixed index logic)')
    # --------------------------------------------------------------------------
    parser.add_argument('--device', type=str, default='cpu', help='cpu only for RPi5')
    parser.add_argument('--model', type=str, default='thunder', choices=['thunder', 'lightning'])
    parser.add_argument('--save_out', type=str, default='', help='Save output video (GUI 제거로 이 기능은 사용하지 않습니다)')
    parser.add_argument('--show_skeleton', action='store_true', help='Show skeleton (GUI 제거로 이 기능은 사용하지 않습니다)')
    args = parser.parse_args()
    
    print("="*60)
    print("Raspberry Pi 5 Optimized Fall Detection System (MQTT PE Client)")
    print("- Pose: MoveNet " + args.model.title())
    print("- Detection: Rule-based")
    print("- Device: CPU")
    print(f"- MQTT Broker: {MQTT_BROKER}:{MQTT_PORT} / Module: {PE_MODULE}")
    print("="*60)
    
    # 1. 모델 로드
    print("\n1️⃣ Loading models...")
    detector = YOLOv8_Detector(model_name='yolov8n.pt', conf_thres=0.5, device='cpu')
    pose_model = MoveNetPose(model_type=args.model)
    
    # 2. 트래커
    tracker = SimpleTracker(max_age=50)
    
    # 3. MQTT 클라이언트 초기화
    mqtt_client = mqtt.Client(client_id="PE_Client", protocol=mqtt.MQTTv311)
    
    # 사용자 이름 및 비밀번호 설정
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[{now_str()}] ❌ Failed to connect to MQTT broker: {e}")
        
    # 4. 카메라 초기화 (수정된 부분 2/2)
    print("\n2️⃣ Opening camera...")
    
    # ============================================================
    # 🚨 카메라 초기화 로직 수정: 고정 인덱스 할당 🚨
    # find_camera_by_vid_pid를 사용하여 PE 카메라의 고정 인덱스를 찾습니다.
    # ============================================================
    print(f"[{now_str()}] INFO System :: PE 카메라 고유 ID 기반 인덱스 검색 중...")
    
    # AD 인덱스는 무시하고, PE 인덱스만 추출합니다.
    _, PE_CAMERA_INDEX = find_camera_by_vid_pid()
    
    if PE_CAMERA_INDEX == -1:
        print(f"[{now_str()}] ❌ CRITICAL: PE 카메라 (VID:PID)를 찾을 수 없습니다. 연결 상태를 확인하거나 camera_init_robust.py의 설정을 확인하세요.")
        mqtt_client.loop_stop()
        return

    print(f"[{now_str()}] ✅ PE 카메라 고정 인덱스 확보: {PE_CAMERA_INDEX}")
    
    # 확보된 인덱스로 카메라를 엽니다.
    cap = cv2.VideoCapture(PE_CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"[{now_str()}] ❌ Cannot open camera at fixed index {PE_CAMERA_INDEX}! Check source or permissions.")
        mqtt_client.loop_stop()
        return
    print(f"[{now_str()}] ✅ Camera opened")

    # 상태 변수
    fall_counters = {}
    zone_timers = {}
    
    # Alert 전송 상태 (중복 전송 방지)
    alert_sent_fall = {}
    alert_sent_zone = {}

    fps_time = time.time()
    frame_count = 0
    
    print(f"[{now_str()}] 3️⃣ Starting detection... (Press Ctrl+C to quit)\n")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"[{now_str()}] End of stream or video.")
                break
            
            frame_count += 1
            h, w = frame.shape[:2]
            current_time = time.time()
            
            # 위험구역 그리기 (GUI 제거로 더미 함수 호출)
            frame = draw_danger_area(frame)
            
            # 사람 검출
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
            
            # 트래킹
            current_tracks = tracker.update(detections)
            current_time = time.time()
            
            # RAW 데이터용 리스트 및 위험구역 경고 리스트
            raw_detections_list = []
            zone_warnings = []
            is_person_detected = len(current_tracks) > 0
            
            # 각 트랙 처리
            for track_id in current_tracks:
                track = tracker.tracks[track_id]
                bbox = track['bbox']
                keypoints_list = track['keypoints']
                
                if len(keypoints_list) < 1: continue
                
                x1, y1, x2, y2 = bbox.astype(int)
                center_x, bottom_y = get_location_details(bbox)
                
                # --- [A] 위험구역 체크 (ZONE) ---
                in_zone = is_in_danger_zone(bbox, w, h)
                
                if in_zone:
                    if track_id not in zone_timers:
                        zone_timers[track_id] = current_time
                        alert_sent_zone[track_id] = False
                    
                    elapsed = current_time - zone_timers[track_id]
                    
                    # 🚨 1. ZONE CRITICAL ALERT (임계치 초과)
                    if elapsed >= ZONE_ALERT_TIME and not alert_sent_zone.get(track_id, False):
                        # ALERT 메시지 발행
                        alert_payload = {
                            "timestamp": now_str(),
                            "module": PE_MODULE,
                            "level": "CRITICAL", # 🚨 CRITICAL 레벨 적용
                            "message": f"🚨 DANGER ZONE CRITICAL: Worker #{track_id} in high-risk area for {elapsed:.1f}s. Immediate removal required.",
                            "details": [{"track_id": track_id, "action": "InDangerZoneCritical", "location": f"({center_x}, {bottom_y})"}]
                        }
                        publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                        alert_sent_zone[track_id] = True
                        print(f"[{now_str()}] 🚨🚨 [CRITICAL ALERT SENT] Zone alert for Worker #{track_id} ({elapsed:.1f}s)")

                    # ⚠️ 2. ZONE WARNING (경고 임계치 접근)
                    elif elapsed >= ZONE_WARNING_TIME and not alert_sent_zone.get(track_id, False): 
                        # CRITICAL 임계치 도달 전까지는 WARNING을 유지
                        print(f"[{now_str()}] ⚠️ [WARNING] Worker #{track_id} in zone for {elapsed:.1f}s.")
                        
                else:
                    if track_id in zone_timers:
                        del zone_timers[track_id]
                        alert_sent_zone[track_id] = False

                # --- [B] 낙상 감지 (FALL) ---
                current_kp = keypoints_list[-1]
                prev_kp = keypoints_list[-2] if len(keypoints_list) >= 2 else None
                
                action_name, confidence, details = detect_fall_rule_based(current_kp, prev_kp)
                
                if track_id not in fall_counters:
                    fall_counters[track_id] = 0
                    alert_sent_fall[track_id] = False
                
                is_fall_action = action_name in ['Fall Down', 'Lying Down'] and confidence >= FALL_CONFIDENCE_THRESHOLD
                
                if is_fall_action:
                    fall_counters[track_id] += 1
                else:
                    fall_counters[track_id] = 0
                    alert_sent_fall[track_id] = False # 정상 상태로 돌아오면 알림 상태 초기화
                
                # 🚨 3. FALL CRITICAL ALERT (낙상 확정)
                if fall_counters[track_id] >= FALL_FRAMES and not alert_sent_fall.get(track_id, False):
                    alert_payload = {
                        "timestamp": now_str(),
                        "module": PE_MODULE,
                        "level": "CRITICAL", # 🚨 CRITICAL 레벨 적용
                        "message": f"🚨 CRITICAL FALL DETECTED: Worker #{track_id} is {action_name.lower()} at location ({center_x}, {bottom_y}). Immediate assistance required.",
                        "details": [{"track_id": track_id, "action": action_name, "confidence": float(confidence), "location": f"({center_x}, {bottom_y})"}]
                    }
                    publish_mqtt_message(mqtt_client, ALERT_TOPIC, alert_payload)
                    alert_sent_fall[track_id] = True
                    print(f"[{now_str()}] 🚨🚨 [CRITICAL ALERT SENT] Fall alert for Worker #{track_id} @ {current_time:.2f}")

                # 4. RAW 데이터 기록
                raw_detections_list.append({
                    "track_id": int(track_id),
                    "object_type": "Person",
                    "action": action_name,
                    "confidence": float(confidence),
                    "x_center": center_x,
                    "y_bottom": bottom_y,
                    "in_danger_zone": in_zone
                })
            
            try:
                # 프레임 압축 (JPEG) 및 Base64 인코딩
                ret_enc, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) 
                
                if ret_enc:
                    jpg_as_text = base64.b64encode(buffer.tobytes())
                    
                    # 새로운 VIDEO 토픽 (PE_VIDEO_TOPIC)으로 발행 (QoS=0)
                    mqtt_client.publish(PE_VIDEO_TOPIC, jpg_as_text, qos=0) 
                    
                    if DEBUG_MODE:
                        # 발행 로그: DEBUG_MODE일 때만 출력
                        print(f"[{now_str()}] [PUB-PE-VIDEO] ✅ Base64 frame sent to {PE_VIDEO_TOPIC} (Size: {len(jpg_as_text)/1024:.1f} KB)")
                
            except Exception as e:
                print(f"[{now_str()}] [ERROR] ❌ PE Video streaming publish failed: {e}")
            # ===============================================================================
            
            # 4. RAW 데이터 발행 (주기적으로)
            if frame_count % RAW_PUBLISH_INTERVAL == 0 and is_person_detected:
                raw_payload = {
                    "timestamp": now_str(),
                    "module": PE_MODULE, # PE 모듈 명시
                    "level": "INFO", # 🚨 INFO 레벨 추가 (정상적인 데이터 흐름)
                    "detections": raw_detections_list,
                    "person_detected": is_person_detected
                }

                # 사람이 감지되지 않은 경우
                if not is_person_detected:
                    raw_payload["message"] = "사람이 감지되지 않음."
                else:
                    # 낙상 중인 사람 수 계산
                    fall_count = sum(1 for d in raw_detections_list if d["action"] == "Fall Down")
                    total_count = len(raw_detections_list)
                    if fall_count > 0:
                        raw_payload["message"] = f"{total_count}명 중 {fall_count}명 낙상 감지됨."
                    else:
                        raw_payload["message"] = f"{total_count}명 감지됨. 이상 없음."

                mqtt_client.publish(RAW_TOPIC, json.dumps(raw_payload, ensure_ascii=False), qos=0)
                
                publish_mqtt_message(mqtt_client, RAW_TOPIC, raw_payload)
                # ⭐️ 시연용 로그: RAW 데이터 발행 ⭐️
                end_time = time.time()
                fps = 1.0 / (end_time - fps_time + 1e-6)
                fps_time = end_time
                print(f"[{now_str()}] [PUB-PE-RAW:INFO] ✅ RAW data sent (Tracks: {len(current_tracks)}) (FPS: {fps:.1f})")

            # 짧은 대기 시간을 주어 CPU 사용량을 낮춥니다.
            time.sleep(0.01) # 약 100 FPS (비디오 처리 시간 제외)

        except KeyboardInterrupt:
            print(f"\n[{now_str()}] [INFO System] Measurement stopped by user (Ctrl+C).")
            break
        except Exception as e:
            print(f"\n[{now_str()}] [ERROR System] An unexpected error occurred: {e}")
            break
    
    # 정리
    if cap.isOpened():
        cap.release()
    
    print("\n" + "="*60)
    print("Program terminated. Closing MQTT connection.")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("="*60)


if __name__ == '__main__':
    # TensorFlow 로그를 억제하여 터미널 출력을 깔끔하게 만듭니다.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    main()
