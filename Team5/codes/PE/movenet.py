"""
라즈베리파이5 최적화 낙상 감지 + 상태 전이 분석
MoveNet Thunder + Rule-based Fall Detection + State Transition

주요 기능:
1. YOLOv8 사람 검출 필터링 강화
2. Rule 기반 자세 분류 (Standing, Sitting, Walking, Lying Down)
3. 상태 전이 분석 (Standing/Walking → Lying Down = Fall Down)
4. 위험구역 모니터링
"""

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

DEBUG_MODE = False

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
SHOW_DANGER_AREA = True
DANGER_AREA_COLOR = (0, 0, 255)

# 낙상 판단 설정
FALL_CONFIDENCE_THRESHOLD = 0.70
FALL_FRAMES = 3
FALL_TRANSITION_TIME = 1.0  # 상태 전이 판단 시간 (초)


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
            print(f"Found local TFLite model: {model_path}")
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.use_tflite = True
                print(f"✅ Loaded TFLite model successfully!")
                return
            except Exception as e:
                print(f"❌ TFLite loading failed: {e}")
    
    def predict(self, frame, bboxes, scores=None):
        """프레임 안의 여러 사람에 대한 자세를 예측합니다."""
        if bboxes is None or len(bboxes) == 0:
            return []

        poses = []
        use_tflite = hasattr(self, 'interpreter')
        if not use_tflite:
            print("❌ ERROR: TFLite interpreter not loaded!")
            return []

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = frame[y1:y2, x1:x2]
            crop_height, crop_width, _ = crop_img.shape
            
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
            
            # 좌표 변환 로직
            if crop_height > crop_width:
                padd_x = (self.input_size - (crop_width * self.input_size / crop_height)) / 2
                padd_y = 0
                scale_x = (self.input_size - 2 * padd_x) / crop_width
                scale_y = self.input_size / crop_height
            else:
                padd_x = 0
                padd_y = (self.input_size - (crop_height * self.input_size / crop_width)) / 2
                scale_x = self.input_size / crop_width
                scale_y = (self.input_size - 2 * padd_y) / crop_height

            keypoints = np.zeros((17, 3), dtype=np.float32)
            
            norm_y = keypoints_with_scores[:, 0]
            norm_x = keypoints_with_scores[:, 1]
            
            keypoints[:, 0] = ((norm_x * self.input_size) - padd_x) / scale_x + x1
            keypoints[:, 1] = ((norm_y * self.input_size) - padd_y) / scale_y + y1
            keypoints[:, 2] = keypoints_with_scores[:, 2]
            
            proposal_score = float(np.mean(keypoints[:, 2]))
            
            poses.append({
                'keypoints': keypoints,
                'proposal_score': proposal_score,
                'bbox': bbox
            })
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


def detect_fall_rule_based(keypoints, prev_keypoints=None):
    """균형잡힌 자세 분류 - Standing과 Sitting 모두 정확하게"""
    
    conf = float(np.mean(keypoints[:, 2]))
    valid_kp = keypoints[keypoints[:, 2] > 0.2]
    
    if len(valid_kp) < 5:
        return 'Unknown', conf * 0.3, {}
    
    # 1. 기본 바운딩 박스 특징
    width = valid_kp[:, 0].max() - valid_kp[:, 0].min()
    height = valid_kp[:, 1].max() - valid_kp[:, 1].min()
    ratio = width / (height + 1e-6)
    
    # 2. 주요 관절 위치 추출
    shoulder_y = [keypoints[i][1] for i in [5, 6] if keypoints[i][2] > 0.2]
    shoulder_x = [keypoints[i][0] for i in [5, 6] if keypoints[i][2] > 0.2]
    
    hip_y = [keypoints[i][1] for i in [11, 12] if keypoints[i][2] > 0.15]
    hip_x = [keypoints[i][0] for i in [11, 12] if keypoints[i][2] > 0.15]
    
    knee_y = [keypoints[i][1] for i in [13, 14] if keypoints[i][2] > 0.15]
    ankle_y = [keypoints[i][1] for i in [15, 16] if keypoints[i][2] > 0.15]
    
    # 3. 거리 계산
    sh_dist = 0
    if len(shoulder_y) > 0 and len(hip_y) > 0:
        sh_dist = abs(np.mean(hip_y) - np.mean(shoulder_y))
    
    hk_dist = 0
    if len(hip_y) > 0 and len(knee_y) > 0:
        hk_dist = abs(np.mean(knee_y) - np.mean(hip_y))
    
    ha_dist = 0
    if len(hip_y) > 0 and len(ankle_y) > 0:
        ha_dist = abs(np.mean(ankle_y) - np.mean(hip_y))
    elif len(hip_y) > 0 and len(knee_y) > 0:
        ha_dist = hk_dist * 1.6
    
    shoulder_width = 0
    if len(shoulder_x) >= 2:
        shoulder_width = abs(max(shoulder_x) - min(shoulder_x))
    
    hip_width = 0
    if len(hip_x) >= 2:
        hip_width = abs(max(hip_x) - min(hip_x))
    
    # 4. 모션
    motion = 0.0
    if prev_keypoints is not None:
        motion = estimate_motion(prev_keypoints, keypoints)
    
    # 5. 비율 특징
    upper_ratio = sh_dist / (height + 1e-6)
    lower_ratio = ha_dist / (height + 1e-6)
    thigh_ratio = hk_dist / (height + 1e-6)
    
    horizontal_spread = (shoulder_width + hip_width) / 2
    vertical_horizontal_ratio = height / (horizontal_spread + 1e-6)
    
    details = {
        "ratio": f"{ratio:.2f}",
        "height": f"{height:.0f}",
        "sh": f"{sh_dist:.0f}",
        "ha": f"{ha_dist:.0f}",
        "upper_r": f"{upper_ratio:.2f}",
        "lower_r": f"{lower_ratio:.2f}",
        "vh_ratio": f"{vertical_horizontal_ratio:.2f}",
        "motion": f"{motion:.1f}"
    }
    
    # 6. 균형잡힌 분류 로직
    
    # 🔴 1순위: 누워있음
    if ratio > 1.1 or vertical_horizontal_ratio < 1.5:
        return 'Lying Down', conf * 0.95, details
    
    # 🔴 2순위: 명확한 걷기
    if motion > 8 and height > 100 and lower_ratio > 0.35:
        return 'Walking', conf * 0.92, details
    
    # 🔴 3순위: 명확한 서있음 (Lower가 매우 높음)
    if lower_ratio >= 0.45 and ratio < 0.50:
        return 'Standing', conf * 0.92, details
    
    # 🔴 4순위: 명확한 앉아있음 (2가지 조건 중 하나)
    # 케이스 A: 하체가 매우 짧고 + 가로 비율 높음
    if lower_ratio < 0.25 and ratio > 0.55:
        return 'Sitting', conf * 0.92, details
    
    # 케이스 B: 하체가 짧고 + 상체 비중 높고 + VH 낮음
    if lower_ratio < 0.32 and upper_ratio > 0.50 and vertical_horizontal_ratio < 5.0:
        return 'Sitting', conf * 0.90, details
    
    # 🔴 5순위: 종합 점수 기반 (균형잡힌 판정)
    sitting_score = 0
    standing_score = 0
    
    # 1. Lower 비율 (가장 중요)
    if lower_ratio < 0.20:
        sitting_score += 4
    elif lower_ratio < 0.30:
        sitting_score += 3
    elif lower_ratio < 0.38:
        sitting_score += 1
    elif lower_ratio >= 0.42:
        standing_score += 3
    else:
        standing_score += 1
    
    # 2. Ratio (가로/세로)
    if ratio > 0.65:
        sitting_score += 3
    elif ratio > 0.55:
        sitting_score += 2
    elif ratio < 0.45:
        standing_score += 2
    
    # 3. Upper 비율
    if upper_ratio > 0.60:
        sitting_score += 2
    elif upper_ratio > 0.50:
        sitting_score += 1
    elif upper_ratio < 0.38:
        standing_score += 1
    
    # 4. VH 비율
    if vertical_horizontal_ratio > 10.0:
        standing_score += 2
    elif vertical_horizontal_ratio > 6.0:
        standing_score += 1
    elif vertical_horizontal_ratio < 4.0:
        sitting_score += 2
    elif vertical_horizontal_ratio < 5.5:
        sitting_score += 1
    
    # 5. 키
    if height < 100:
        sitting_score += 2
    elif height > 180:
        standing_score += 1
    
    # 최종 판정 (마진 1로 균형)
    if sitting_score > standing_score + 1:
        return 'Sitting', conf * 0.75, details
    elif standing_score > sitting_score + 1:
        return 'Standing', conf * 0.75, details
    else:
        # 동점이거나 차이가 1이면 Lower 우선
        if lower_ratio < 0.35:
            return 'Sitting', conf * 0.70, details
        else:
            return 'Standing', conf * 0.70, details


# ============================================
# 위험구역 함수
# ============================================

def is_in_danger_zone(bbox, frame_width, frame_height):
    """바운딩박스가 위험구역에 있는지 확인"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    bottom_y = y2
    
    if USE_RATIO:
        x_min = int(frame_width * DANGER_X_RATIO_MIN) if DANGER_X_RATIO_MIN is not None else None
        x_max = int(frame_width * DANGER_X_RATIO_MAX) if DANGER_X_RATIO_MAX is not None else None
        y_min = int(frame_height * DANGER_Y_RATIO_MIN) if DANGER_Y_RATIO_MIN is not None else None
        y_max = int(frame_height * DANGER_Y_RATIO_MAX) if DANGER_Y_RATIO_MAX is not None else None
    else:
        x_min = DANGER_X_MIN
        x_max = DANGER_X_MAX
        y_min = DANGER_Y_MIN
        y_max = DANGER_Y_MAX
    
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


def draw_danger_area(frame):
    """위험구역 시각화"""
    if not SHOW_DANGER_AREA:
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


def draw_zone_warnings(frame, zone_warnings):
    """여러 위험구역 경고를 순차적으로 표시"""
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
# 개선된 트래커 (사람 필터링 강화)
# ============================================

class ImprovedTracker:
    """개선된 IoU 기반 트래커 with 사람 검증"""
    
    def __init__(self, max_age=12):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.frame_count = 0
    
    def _is_valid_person(self, keypoints, bbox):
        """사람인지 검증하는 추가 필터"""
        # 1. 평균 키포인트 신뢰도
        avg_conf = np.mean(keypoints[:, 2])
        if avg_conf < 0.25:
            return False
        
        # 2. 주요 관절(어깨, 엉덩이) 존재 여부
        important_kp = [5, 6, 11, 12]
        important_conf = [keypoints[i][2] for i in important_kp if i < len(keypoints)]
        if len(important_conf) < 2 or np.mean(important_conf) < 0.3:
            return False
        
        # 3. bbox 크기 체크
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width < 20 or height < 30:
            return False
        if width > 700 or height > 800:
            return False
        
        # 4. 종횡비 체크
        aspect = height / (width + 1e-6)
        if aspect < 0.4:
            return False
        
        return True
    
    def update(self, detections):
        """detections: List of {'bbox': [x1,y1,x2,y2], 'keypoints': array, 'score': float}"""
        self.frame_count += 1
        
        # 기존 트랙 나이 증가
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                print(f"[TRACKER] Removing track #{track_id} (age: {self.tracks[track_id]['age']})")
                del self.tracks[track_id]
        
        # 사람 검증 필터링 추가
        valid_detections = []
        for det in detections:
            if self._is_valid_person(det['keypoints'], det['bbox']):
                valid_detections.append(det)
            elif DEBUG_MODE:
                print(f"  Filtered out invalid detection (low quality)")
        
        # IoU 매칭
        for det in valid_detections:
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
                self.tracks[best_id]['age'] = 0
            else:
                # 새 트랙 생성
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'keypoints': [det['keypoints']],
                    'age': 0
                }
                self.next_id += 1
        
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

class ImprovedYOLODetector:
    """강화된 사람 검출 필터링"""
    
    def __init__(self, model_name='yolov8n.pt', conf_thres=0.5, device='cpu'):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        self.device = device
    
    def detect(self, frame):
        """사람만 검출 (추가 필터링)"""
        #프레임 리사이즈(YOLO입력만)
        h,w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (640,480))

        results = self.model.predict(
            frame,
            conf=self.conf_thres,
            classes=[0],
            device=self.device,
            verbose=False,
            imgsz=320
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return [], []
        
        boxes = results[0].boxes
        bboxes = []
        scores = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            if cls != 0:
                continue
            
            width = x2 - x1
            height = y2 - y1
            
            if width < 40 or height < 35:
                if DEBUG_MODE:
                    print(f"  Filtered bbox: too small ({width:.0f}x{height:.0f})")
                continue
            
            if width > frame.shape[1] * 0.9 or height > frame.shape[0] * 0.9:
                if DEBUG_MODE:
                    print(f"  Filtered bbox: too large ({width:.0f}x{height:.0f})")
                continue
            
            aspect = height / (width + 1e-6)
            if aspect < 0.3:
                if DEBUG_MODE:
                    print(f"  Filtered bbox: invalid aspect ratio {aspect:.2f}")
                continue
            
            bboxes.append([x1, y1, x2, y2])
            scores.append(conf)
        
        return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)


# ============================================
# 스켈레톤 그리기
# ============================================

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_skeleton(frame, keypoints):
    """스켈레톤 그리기"""
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3:
            start = tuple(keypoints[start_idx][:2].astype(int))
            end = tuple(keypoints[end_idx][:2].astype(int))
            cv2.line(frame, start, end, (0, 255, 0), 2)
    
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.3:
            cv2.circle(frame, tuple(kp[:2].astype(int)), 4, (0, 0, 255), -1)
    
    return frame


# ============================================
# 메인 함수
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Improved Fall Detection with State Transition')
    parser.add_argument('--camera', type=str, default='0', help='Camera source')
    parser.add_argument('--device', type=str, default='cpu', help='cpu only for RPi5')
    parser.add_argument('--model', type=str, default='thunder', choices=['thunder', 'lightning'])
    parser.add_argument('--save_out', type=str, default='', help='Save output video')
    parser.add_argument('--show_skeleton', action='store_true', help='Show skeleton')
    args = parser.parse_args()
    
    print("="*60)
    print("Improved Fall Detection System")
    print("- Object Detection: YOLOv8n (Enhanced Filtering)")
    print("- Pose: MoveNet " + args.model.title())
    print("- Device: CPU")
    print("="*60)
    
    # 모델 로드
    print("\n1️⃣ Loading models...")
    detector = ImprovedYOLODetector(model_name='yolov8n.pt', conf_thres=0.5, device='cpu')
    pose_model = MoveNetPose(model_type=args.model)
    
    # 트래커
    tracker = ImprovedTracker(max_age=12)
    
    # 카메라
    print("\n2️⃣ Opening camera...")
    cam_source = args.camera
    if cam_source.isdigit():
        cap = cv2.VideoCapture(int(cam_source))
    else:
        cap = cv2.VideoCapture(cam_source)
    
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return
    
    print("✅ Camera opened")
    
    # 비디오 저장
    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_out, fourcc, fps, (width, height))
    
    # 상태 변수
    fall_counters = {}
    fall_alerted = {}  # 🔴 낙상 알림 여부 추적
    zone_timers = {}
    previous_states = {}  # 🔴 상태 전이 분석용
    
    fps_time = time.time()
    frame_count = 0
    
    print("\n3️⃣ Starting detection... (Press 'q' to quit)\n")
   
    last_bboxes = []
    last_scores = []
    yolo_skip_counter = 0
    YOLO_SKIP_FRAMES =3
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # 위험구역 그리기
        frame = draw_danger_area(frame)

        yolo_skip_counter += 1
        if yolo_skip_counter >= YOLO_SKIP_FRAMES:
            bboxes, scores = detector.detect(frame)
            last_bboxes = bboxes
            last_scores = scores
            yolo_skip_counter = 0
        else: 
            bboxes = last_bboxes
            scores = last_scores        
        # 사람 검출
        bboxes, scores = detector.detect(frame)
        
        if DEBUG_MODE and frame_count % 30 == 0:
            print(f"\nFrame {frame_count}: Detected {len(bboxes)} persons")
        
        # 포즈 추정
        detections = []
        if len(bboxes) > 0:
            poses = pose_model.predict(frame, bboxes, scores)
            
            for pose, bbox in zip(poses, bboxes):
                keypoints = pose['keypoints']
                
                detections.append({
                    'bbox': bbox,
                    'keypoints': keypoints,
                    'score': pose['proposal_score']
                })
        
        # 트래킹
        current_tracks = tracker.update(detections)
        current_time = time.time()
        
        # 위험구역 경고 리스트
        zone_warnings = []
        
        # 각 트랙 처리
        for track_id in current_tracks:
            track = tracker.tracks[track_id]
            bbox = track['bbox']
            keypoints_list = track['keypoints']
            if len(keypoints_list) < 1:
                continue
            
            x1, y1, x2, y2 = bbox.astype(int)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # 위험구역 체크
            in_zone = is_in_danger_zone(bbox, w, h)
            
            if in_zone:
                if track_id not in zone_timers:
                    zone_timers[track_id] = current_time
                    print(f"[DANGER ZONE] Worker #{track_id} entered!")
                
                elapsed = current_time - zone_timers[track_id]
                
                time_text = f"Zone: {elapsed:.1f}s"
                cv2.putText(frame, time_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if elapsed >= ZONE_ALERT_TIME:
                    zone_warnings.append((track_id, elapsed, 'danger'))
                elif elapsed >= ZONE_WARNING_TIME:
                    zone_warnings.append((track_id, elapsed, 'warning'))
            else:
                if track_id in zone_timers:
                    final_time = current_time - zone_timers[track_id]
                    print(f"[DANGER ZONE] Worker #{track_id} left (stayed {final_time:.1f}s)")
                    del zone_timers[track_id]
            
            # 🔴 ========================================
            # Rule 기반 자세 분류
            # 🔴 ========================================
            current_kp = keypoints_list[-1]
            prev_kp = keypoints_list[-2] if len(keypoints_list) >= 2 else None
            
            rule_action, rule_conf, details = detect_fall_rule_based(current_kp, prev_kp)
            
            # 🔴 ========================================
            # 상태 전이 분석: Fall Down 감지
            # (Rule 결과 직후에 수행)
            # 🔴 ========================================
            
            # 이전 상태 가져오기
            if track_id in previous_states:
                prev_action = previous_states[track_id]['action']
                state_start_time = previous_states[track_id]['state_start_time']
                time_since_state_start = current_time - state_start_time
            else:
                prev_action = None
                state_start_time = current_time
                time_since_state_start = 0
            
            # 낙상 전이 감지: (Standing/Walking) → Lying Down
            final_action = rule_action
            final_conf = rule_conf
            new_state_start_time = state_start_time  # 기본값: 이전 시작 시간 유지
            
            # 🔴 케이스 1: Lying Down은 무조건 Fall Down으로 변경!
            if rule_action == 'Lying Down':
                final_action = 'Fall Down'
                final_conf = rule_conf * 1.15
                
                # 처음 누운 것이면 새로운 시작 시간 기록
                if prev_action != 'Fall Down':
                    new_state_start_time = current_time
                    if DEBUG_MODE:
                        print(f"\n🚨 [FALL DETECTED!] Worker #{track_id}")
                        print(f"   Transition: {prev_action} → Fall Down")
                # 이미 Fall Down 상태면 시간 유지
                else:
                    # new_state_start_time 유지
                    if DEBUG_MODE and frame_count % 30 == 0:
                        print(f"[FALL SUSTAINED] Worker #{track_id} in Fall Down ({time_since_state_start:.1f}s)")
            
            # 🔴 케이스 2: Standing/Walking → Lying Down (빠른 전환 감지는 제거)
            # (위에서 이미 처리됨)
            
            # 🔴 케이스 2: Fall Down → Standing/Walking/Sitting (회복)
            elif prev_action == 'Fall Down' and rule_action in ['Standing', 'Walking', 'Sitting']:
                final_action = rule_action
                new_state_start_time = current_time  # 새로운 상태 시작!
                if DEBUG_MODE:
                    print(f"[RECOVERY] Worker #{track_id} recovered from Fall Down → {rule_action}")
            
            # 🔴 케이스 3: 상태 변경
            elif prev_action != rule_action:
                final_action = rule_action
                new_state_start_time = current_time  # 새로운 상태 시작!
            
            # 🔴 케이스 4: 상태 유지 (timestamp 변경 없음)
            else:
                final_action = rule_action
                # new_state_start_time 유지
            
            # 🔴 현재 상태 저장
            previous_states[track_id] = {
                'action': final_action,
                'state_start_time': new_state_start_time  # 상태 시작 시간 추적
            }
            
            # 디버그 출력
            if DEBUG_MODE and frame_count % 10 == 0:
                print(f"[DEBUG] ID:{track_id} | Final:'{final_action}'({final_conf:.2f}) | "
                      f"Prev:'{prev_action}' | "
                      f"Rule:'{rule_action}' | "
                      f"H:{details.get('height', 'N/A')} | "
                      f"R:{details.get('ratio', 'N/A')} | "
                      f"Upper:{details.get('upper_r', 'N/A')} | "
                      f"Lower:{details.get('lower_r', 'N/A')} | "
                      f"VH:{details.get('vh_ratio', 'N/A')} | "
                      f"M:{details.get('motion', 'N/A')}")
            
            # 낙상 카운터
            if track_id not in fall_counters:
                fall_counters[track_id] = 0
            if track_id not in fall_alerted:
                fall_alerted[track_id] = False
            
            if final_action in ['Fall Down'] and final_conf >= FALL_CONFIDENCE_THRESHOLD:
                fall_counters[track_id] += 1
            else:
                fall_counters[track_id] = 0
                fall_alerted[track_id] = False  # 정상 상태로 돌아오면 알림 리셋
            
            # 상태 및 색상 결정
            if fall_counters[track_id] >= FALL_FRAMES:
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (0, 0, 255)  # 🔴 확정 낙상
                
                # 🚨 확정 낙상 메시지 출력 (딱 한 번만!)
                if not fall_alerted[track_id]:
                    print(f"\n🚨 [FALL DETECTED!] Worker #{track_id} - {final_action}")
                    print(f"   Confidence: {final_conf*100:.1f}%")
                    print(f"   Fall frames: {fall_counters[track_id]}")
                    fall_alerted[track_id] = True
                    
            elif final_action == 'Fall Down':
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (0, 0, 255)  # 🔴 Fall Down (아직 확정 전)
            elif final_action == 'Standing':
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (0, 255, 0)  # 🟢 정상
            elif final_action == 'Sitting':
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (0, 255, 255)  # 🟡 앉음
            elif final_action == 'Walking':
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (255, 255, 0)  # 하늘색
            else:
                action = f'{final_action}: {final_conf*100:.1f}%'
                clr = (255, 255, 255)  # ⚪ Unknown
            
            # 시각화
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', center,
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, action, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_COMPLEX, 0.5, clr, 2)
            
            # 스켈레톤
            if args.show_skeleton:
                frame = draw_skeleton(frame, current_kp)
        
        # 위험구역 경고 표시
        if zone_warnings:
            frame = draw_zone_warnings(frame, zone_warnings)
        
        # FPS 표시
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        
        info_text = f'Frame: {frame_count} | FPS: {fps:.1f} | Persons: {len(current_tracks)}'
        cv2.putText(frame, info_text,
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 화면 표시
        cv2.imshow('Improved Fall Detection', frame)
        
        # 비디오 저장
        if writer:
            writer.write(frame)
        
        # 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 정리
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("Program terminated")
    print(f"Total frames processed: {frame_count}")
    print("="*60)


if __name__ == '__main__':
    main()
