
import torch
import torch.nn as nn
import math

class PatchTSTCollisionPredictor(nn.Module):
    def __init__(self, c_in, seq_len, patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1, n_classes=2):
        super().__init__()
        n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=stride)
        pe = torch.zeros(n_patches, d_model)
        pos = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_encoding', pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * n_patches, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        x = self.patch_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x + self.pos_encoding)
        x = self.encoder(x)
        return self.classifier(x)
#!/usr/bin/env python

"""
CARLA Manual Control + Crash Detection 모듈 통합
- CARLA 원본 manual_control.py를 모듈로 import
- Crash Detection 모델만 추가
- 기존 UI에 위험도 정보 표시
"""

import sys
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import pickle
from collections import deque
import time
import cv2

# YOLO 모델 로드를 위한 경로 추가
sys.path.append('yolov7')

CARLA_ROOT = r"/home/jeongseon/carla"
CARLA_EXAMPLES_PATH = os.path.join(CARLA_ROOT, "PythonAPI", "examples")

sys.path.append(CARLA_EXAMPLES_PATH)
try:
    import importlib.util as _importlib_util
    manual_control_path = os.path.join(CARLA_EXAMPLES_PATH, "manual_control.py")
    if os.path.exists(manual_control_path):
        spec = _importlib_util.spec_from_file_location("carla_examples_manual_control", manual_control_path)
        manual_control = _importlib_util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(manual_control)
        print("✅ CARLA 0.9.16 manual_control 예제 로드 성공")
    else:
        raise FileNotFoundError(f"manual_control.py not found at {manual_control_path}")
except Exception as e:
    print(f"❌ CARLA manual_control 로드 실패: {e}")
    sys.exit(1)

class YOLODetector:
    """YOLO 객체 감지 모듈"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_yolo_model()
        
    def load_yolo_model(self):
        """YOLO 모델 로드"""
        try:
            # YOLOv7 모델 로드
            model_path = 'yolov7/yolov7.pt'
            if os.path.exists(model_path):
                from models.experimental import attempt_load
                from utils.general import non_max_suppression, scale_coords
                from utils.torch_utils import select_device
                
                self.model = attempt_load(model_path, map_location=self.device)
                self.model.eval()
                print("✅ YOLOv7 모델 로드 완료")
            else:
                print("⚠️ YOLOv7 모델 파일을 찾을 수 없습니다. YOLO 감지 기능 비활성화.")
                self.model = None
        except Exception as e:
            print(f"❌ YOLOv7 모델 로드 실패: {e}. YOLO 감지 기능 비활성화.")
            self.model = None
    
    def detect_objects(self, image):
        """객체 감지 실행"""
        if self.model is None:
            return []
        
        try:
            from utils.general import non_max_suppression, scale_coords
            
            # 이미지 전처리
            img = cv2.resize(image, (640, 640))
            img = img[:, :, ::-1].transpose(2, 0, 1) 
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 추론
            with torch.no_grad():
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, 0.25, 0.45)
            
            detections = []
            if pred[0] is not None:
                for *xyxy, conf, cls in pred[0]:
                    if int(cls) == 2:  # 차량(car)만 감지
                        detections.append({
                            'bbox': [int(x) for x in xyxy],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
            
            return detections
        except Exception as e:
            return []

class CrashDetectionModule:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 60
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        # YOLO 감지기 초기화
        self.yolo_detector = YOLODetector()
        
        # ResNet18 모델 로드
        self.resnet_model = None
        self.load_resnet_model()
        
        # 훈련된 모델 로드 시도
        self.scaler = None
        self.trained_model = None
        self.load_scaler()
        self.load_trained_model()
        
        # 충돌 확률
        self.crash_probability = 0.0
        
        # 분석 주기 제어 (매 N 프레임마다 분석)
        self.analysis_interval = 10 
        self.frame_count = 0
        self.last_analysis_time = 0
        
        # YOLO 감지 결과 저장
        self.detected_objects = []
        
        print(f"CrashDetectionModule 초기화 완료 - Device: {self.device}")

    def load_resnet_model(self):
        """ResNet18 특징 추출 모델 로드"""
        try:
            self.resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.resnet_model.fc = nn.Identity()
            self.resnet_model.to(self.device)
            self.resnet_model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"❌ ResNet18 모델 로드 실패: {e}")
            self.resnet_model = None

    def load_trained_model(self):
        """훈련된 충돌 예측 모델 로드 (커스텀 PatchTSTCollisionPredictor 사용)"""
        try:
            model_path = './model/best_model_front_classification.pth'
            config_path = './model/scaler_front_classification.pkl'
            # 모델 config 정보는 직접 지정하거나 scaler 파일에서 추출
            model_config = {
                'c_in': 16, # 입력 feature 개수 (학습 코드 SequenceGenerator.features와 동일)
                'seq_len': 60, # 시퀀스 길이
                'patch_len': 16,
                'stride': 8,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 3,
                'd_ff': 256,
                'dropout': 0.1,
                'n_classes': 2
            }
            if os.path.exists(model_path):
                try:
                    model = PatchTSTCollisionPredictor(**model_config).to(self.device)
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint)
                    model.eval()
                    self.trained_model = model
                    print("✅ 훈련된 PatchTSTCollisionPredictor 모델 로드 완료")
                except Exception as model_error:
                    print(f"⚠️ PatchTSTCollisionPredictor 모델 로드 실패: {model_error}. 규칙 기반 분석을 사용합니다.")
                    self.trained_model = None
            else:
                print("⚠️ 훈련된 모델 파일을 찾을 수 없습니다. 규칙 기반 분석 사용.")
                self.trained_model = None
        except Exception as e:
            print(f"❌ 훈련된 모델 로드 실패: {e}. 규칙 기반 분석 사용.")
            self.trained_model = None

    def load_scaler(self):
        """스케일러 로드"""
        try:
            # 우선 model/ 경로를 확인하고, 없으면 model_online/ 경로 확인
            scaler_path_candidates = [
                './model/scaler_front_classification.pkl',
            ]
            
            loaded = False
            for scaler_path in scaler_path_candidates:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"✅ Scaler 로드 완료: {scaler_path}")
                    loaded = True
                    break
            if not loaded:
                print("⚠️ Scaler 파일을 찾을 수 없습니다.")
                self.scaler = None
        except Exception as e:
            print(f"❌ Scaler 로드 실패: {e}")
            self.scaler = None

    def extract_visual_features(self, image_array):
        """시각적 특징 추출"""
        if self.resnet_model is None:
            return [0] * 512
        
        try:
            # numpy array를 PIL Image로 변환
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_rgb = image_array[:, :, :3]  # RGBA to RGB
            else:
                image_rgb = image_array
            
            # PyTorch 텐서로 변환
            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet_model(image_tensor)
                features = features.cpu().numpy().flatten()
            
            return features.tolist()
        except Exception as e:
            print(f"❌ 시각적 특징 추출 오류: {e}")
            return [0] * 512

    def calculate_crash_probability(self, world):
        """충돌 확률 계산 - 주기적으로만 실행"""
        if not world.player:
            return self.crash_probability
        
        # 프레임 카운트 증가 (버퍼는 매 프레임 채움, 무거운 분석만 주기적으로)
        self.frame_count += 1
        current_time = time.time()
        do_heavy = (self.frame_count % self.analysis_interval == 0) and (current_time - self.last_analysis_time >= 0.1)
        if do_heavy:
            self.last_analysis_time = current_time

        try:
            # 차량 상태 정보 가져오기
            velocity = world.player.get_velocity()
            speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            control = world.player.get_control()

            # 현재 카메라 이미지 가져오기 (가볍게)
            current_image = None
            if do_heavy and world.camera_manager and hasattr(world.camera_manager, 'surface') and world.camera_manager.surface:
                try:
                    import pygame
                    # pygame surface를 numpy array로 변환 (매 10프레임마다만)
                    current_image = pygame.surfarray.array3d(world.camera_manager.surface)
                    current_image = np.transpose(current_image, (1, 0, 2))
                except Exception as e:
                    current_image = None

            # YOLO 객체 감지 실행 (원래대로)
            if current_image is not None and (self.frame_count % 20 == 0):
                self.detected_objects = self.yolo_detector.detect_objects(current_image)


            # === PatchTST 입력용 16개 feature 추출 (YOLO 감지 차량별) ===
            # features = ['relative_x', 'relative_y', 'relative_vx', 'relative_vy', 'relative_speed', 'ego_speed', 'ego_acceleration', 'vehicle_speed', 'vehicle_acceleration', 'ego_distance', 'min_distance', 'approach_rate', 'ttc', 'distance_change_rate', 'collision_probability', 'critical_distance']
            # ego 차량 정보
            ego_transform = world.player.get_transform()
            ego_velocity = world.player.get_velocity()
            ego_speed = (ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)**0.5
            ego_x, ego_y = ego_transform.location.x, ego_transform.location.y
            ego_vx, ego_vy = ego_velocity.x, ego_velocity.y
            ego_yaw = ego_transform.rotation.yaw
            # TODO: ego_acceleration, 추정치로 0 사용
            ego_acceleration = 0.0

            # YOLO 감지 차량별 위험도 예측
            if self.detected_objects:
                for det in self.detected_objects:
                    # bbox 중심을 차량 위치로 가정 (실제 차량 위치와 다를 수 있음)
                    x1, y1, x2, y2 = det['bbox']
                    vehicle_x = (x1 + x2) / 2
                    vehicle_y = (y1 + y2) / 2
                    # 차량 속도/가속도 정보는 알 수 없으므로 0으로 대체
                    vehicle_vx, vehicle_vy = 0.0, 0.0
                    vehicle_speed = 0.0
                    vehicle_acceleration = 0.0
                    # 상대 위치/속도
                    relative_x = vehicle_x - ego_x
                    relative_y = vehicle_y - ego_y
                    relative_vx = vehicle_vx - ego_vx
                    relative_vy = vehicle_vy - ego_vy
                    relative_speed = (relative_vx**2 + relative_vy**2)**0.5
                    # 거리
                    ego_distance = (relative_x**2 + relative_y**2)**0.5
                    min_distance = ego_distance
                    # approach_rate, ttc, distance_change_rate, collision_probability, critical_distance: 추정치로 0 사용
                    approach_rate = 0.0
                    ttc = 999.0
                    distance_change_rate = 0.0
                    collision_probability = 0.0
                    critical_distance = 0.0
                    # feature vector
                    patchtst_features = [
                        relative_x, relative_y, relative_vx, relative_vy, relative_speed,
                        ego_speed, ego_acceleration, vehicle_speed, vehicle_acceleration,
                        ego_distance, min_distance, approach_rate, ttc, distance_change_rate,
                        collision_probability, critical_distance
                    ]
                    self.feature_buffer.append(patchtst_features)
            else:
                # 감지 차량 없으면 0벡터
                patchtst_features = [0.0] * 16
                self.feature_buffer.append(patchtst_features)

            # 무거운 분석 주기일 때만 위험도 업데이트
            if do_heavy:
                if self.trained_model is not None and len(self.feature_buffer) >= self.sequence_length:
                    risk_score = self.predict_with_model()
                else:
                    # 휴리스틱 기반 동적 위험도
                    risk_score = self._heuristic_risk(speed_kmh, control, self.detected_objects)
                self.crash_probability = max(0.0, min(1.0, risk_score))
            return self.crash_probability

        except Exception as e:
            return self.crash_probability

    def predict_with_model(self):
        """훈련된 PatchTST 모델로 충돌 확률 예측"""
        try:
            # 시퀀스 데이터 준비
            sequence_data = np.array(list(self.feature_buffer))
            
            # 스케일링
            if self.scaler is not None:
                # 각 프레임별로 스케일링
                scaled_sequence = []
                for frame in sequence_data:
                    scaled_frame = self.scaler.transform([frame])[0]
                    scaled_sequence.append(scaled_frame)
                sequence_data = np.array(scaled_sequence)
            
            # PatchTST 입력 형태로 변환: (batch_size, sequence_length, num_features)
            input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
            # 모델 예측
            with torch.no_grad():
                outputs = self.trained_model(input_tensor)
                # outputs: (batch_size, n_classes)
                if torch.is_tensor(outputs):
                    probabilities = torch.softmax(outputs, dim=-1)
                    crash_prob = probabilities[0][1].item()  # 충돌 클래스 확률
                else:
                    print(f"PatchTSTCollisionPredictor 출력 타입: {type(outputs)}")
                    return 0.1
            return crash_prob
            
        except Exception as e:
            print(f"❌ PatchTST 모델 예측 오류: {e}")
            return 0.1  # AI 모델 실패 시 낮은 위험도 반환

    def get_risk_level_and_color(self):
        """위험도 레벨과 색상 반환"""
        if self.crash_probability > 0.7:
            return "CRITICAL", (255, 0, 0)
        elif self.crash_probability > 0.5:
            return "HIGH", (255, 165, 0)
        elif self.crash_probability > 0.3:
            return "MEDIUM", (255, 255, 0)
        else:
            return "LOW", (0, 255, 0)

    def _heuristic_risk(self, speed_kmh, control, detections):
        """모델 미사용/워밍업 시 동적 위험도 계산 휴리스틱"""
        try:
            # 속도 기반 (최대 0.4 가중)
            speed_term = min(speed_kmh / 120.0, 1.0) * 0.4

            # 조향/스로틀 기반 (고속에서 급조향/급가속 위험)
            steer_term = min(abs(control.steer), 1.0)
            throttle_term = min(max(control.throttle, 0.0), 1.0)
            dynamic_term = min(speed_kmh / 80.0, 1.0) * (0.3 * steer_term + 0.2 * throttle_term)

            # 브레이크는 위험 감소
            brake_term = min(max(control.brake, 0.0), 1.0)
            brake_reduction = 0.2 * brake_term

            # YOLO 근접도(박스 면적 비례) 기반 (최대 0.5 가중)
            proximity = 0.0
            if detections:
                areas = []
                for d in detections:
                    x1, y1, x2, y2 = d.get('bbox', [0, 0, 0, 0])
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    area_ratio = (w * h) / float(640 * 640)
                    score = area_ratio * float(d.get('confidence', 0.0))
                    areas.append(score)
                areas.sort(reverse=True)
                proximity = sum(areas[:3])  # 상위 3개 합
                proximity = min(proximity * 3.0, 1.0)  # 스케일링 및 클램프
            proximity_term = 0.5 * proximity

            risk = speed_term + dynamic_term + proximity_term - brake_reduction
            return float(max(0.0, min(1.0, risk)))
        except Exception:
            return 0.1

# CrashDetectionModule 인스턴스 생성
crash_detector = CrashDetectionModule()

class SpeedLimitedKeyboardControl(manual_control.KeyboardControl):
    """속도 제한이 있는 키보드 컨트롤"""
    
    def __init__(self, world, autopilot_enabled=False, max_speed_kmh=60):
        super().__init__(world, autopilot_enabled)
        self.max_speed_kmh = max_speed_kmh
        print(f"🚗 속도 제한 설정: {max_speed_kmh} km/h")
    
    def parse_events(self, client, world, clock, sync_mode):
        """원본 이벤트 파싱에 속도 제한 추가"""
        # 원본 이벤트 파싱 실행
        result = super().parse_events(client, world, clock, sync_mode)
        
        # 속도 제한 적용
        if world.player is not None:
            self.apply_speed_limit(world.player)
        
        return result
    
    def apply_speed_limit(self, vehicle):
        """차량에 속도 제한 적용"""
        try:
            # 현재 속도 확인
            velocity = vehicle.get_velocity()
            current_speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            
            # 속도 제한 초과 시 브레이크 적용
            if current_speed_kmh > self.max_speed_kmh:
                control = vehicle.get_control()
                # 속도 초과량에 따라 브레이크 강도 조절
                speed_excess = current_speed_kmh - self.max_speed_kmh
                brake_force = min(0.8, speed_excess / 20.0)  # 최대 0.8까지 브레이크
                
                control.brake = brake_force
                control.throttle = 0.0  # 스로틀 차단
                vehicle.apply_control(control)
                
        except Exception as e:
            pass  # 에러 무시

# 원본 HUD 클래스를 확장
class CrashDetectionHUD(manual_control.HUD):
    """원본 HUD를 확장하여 crash detection 정보 추가"""
    
    def __init__(self, width, height, sim_speed=1.0, target_fps=20):
        super().__init__(width, height)
        self.crash_detector = crash_detector
        self.sim_speed = sim_speed
        self.target_fps = target_fps

    def tick(self, world, clock):
        """원본 tick 메서드를 확장"""
        # 원본 tick 실행
        super().tick(world, clock)
        
        # crash detection 실행
        crash_prob = self.crash_detector.calculate_crash_probability(world)
        risk_level, risk_color = self.crash_detector.get_risk_level_and_color()
        
        # 현재 속도 정보 추가
        current_speed = 0.0
        if world.player:
            velocity = world.player.get_velocity()
            current_speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        
        # YOLO 감지 차량 수 계산
        detected_cars = len(self.crash_detector.detected_objects) if self.crash_detector.detected_objects else 0
        high_conf_cars = sum(1 for obj in self.crash_detector.detected_objects if obj['confidence'] > 0.5) if self.crash_detector.detected_objects else 0
        
        # 모델 상태 확인 - AI 모델만 사용
        if self.crash_detector.trained_model is not None:
            if len(self.crash_detector.feature_buffer) >= self.crash_detector.sequence_length:
                model_status = "AI PREDICTION"
                model_color = "🤖"
            else:
                model_status = "AI LOADING..."
                model_color = "⏳"
        else:
            model_status = "AI UNAVAILABLE"
            model_color = "❌"
        
        # HUD 정보에 crash detection + 속도 + YOLO + 시뮬레이션 정보 추가
        crash_info = [
            '',
            '🚨 CRASH DETECTION 🚨',
            f'Risk Level: {risk_level}',
            f'Probability: {crash_prob*100:.1f}%',
            f'Model: {model_color} {model_status}',
            '',
            '🚗 CAR DETECTION 🚗',
            f'Cars Detected: {detected_cars}',
            f'High Confidence: {high_conf_cars}',
            '',
            '🏎️ SPEED CONTROL 🏎️',
            f'Current Speed: {current_speed:.1f} km/h',
            f'Speed Limit: 60 km/h',
            '',
            '⚙️ SIMULATION ⚙️',
            f'Sim Speed: {self.sim_speed}x',
            f'Target FPS: {self.target_fps}',
            f'Buffer: {len(self.crash_detector.feature_buffer)}/{self.crash_detector.sequence_length}',
            f'Device: {str(self.crash_detector.device).upper()}'
        ]
        
        # 기존 정보 앞에 crash detection 정보 삽입
        self._info_text = crash_info + self._info_text

    def render(self, display):
        """원본 render 메서드를 확장"""
        # 원본 render 실행
        super().render(display)
        
        # 추가적인 crash risk 시각적 표시기
        self.render_crash_indicator(display)

    def render_crash_indicator(self, display):
        """화면 상단에 큰 crash risk 표시기 + 속도계 + YOLO 감지 박스"""
        try:
            import pygame
            
            # 위험도 정보
            risk_level, risk_color = self.crash_detector.get_risk_level_and_color()
            crash_prob = self.crash_detector.crash_probability
            
            # 현재 속도 정보
            current_speed = 0.0
            if hasattr(self, '_info_text') and len(self._info_text) > 6:
                # HUD에서 속도 정보 추출
                for line in self._info_text:
                    if 'Current Speed:' in str(line):
                        try:
                            speed_str = str(line).split(':')[1].split('km/h')[0].strip()
                            current_speed = float(speed_str)
                        except:
                            current_speed = 0.0
                        break
            
            # === YOLO 감지 결과를 화면에 표시 ===
            self.render_yolo_detections(display)
            
            # 표시기 위치 및 크기 (더 큰 패널)
            indicator_width = 400
            indicator_height = 80
            indicator_x = (self.dim[0] - indicator_width) // 2
            indicator_y = 10
            
            # 배경 패널
            panel_surface = pygame.Surface((indicator_width, indicator_height))
            panel_surface.set_alpha(180)
            panel_surface.fill((0, 0, 0))
            display.blit(panel_surface, (indicator_x, indicator_y))
            
            # === 충돌 위험도 섹션 ===
            # 제목
            title_font = pygame.font.Font(None, 20)
            title_surface = title_font.render("CRASH RISK", True, (255, 255, 255))
            display.blit(title_surface, (indicator_x + 10, indicator_y + 8))
            
            # 위험도 퍼센트
            percent_font = pygame.font.Font(None, 28)
            percent_text = f"{crash_prob*100:.0f}%"
            percent_surface = percent_font.render(percent_text, True, risk_color)
            display.blit(percent_surface, (indicator_x + 100, indicator_y + 5))
            
            # 위험도 레벨
            level_font = pygame.font.Font(None, 16)
            level_surface = level_font.render(risk_level, True, risk_color)
            display.blit(level_surface, (indicator_x + 10, indicator_y + 30))
            
            # === 속도계 섹션 ===
            # 속도 제목
            speed_title = title_font.render("SPEED", True, (255, 255, 255))
            display.blit(speed_title, (indicator_x + 10, indicator_y + 50))
            
            # 현재 속도
            speed_text = f"{current_speed:.0f} km/h"
            speed_color = (255, 255, 255)
            if current_speed > 60:  # 속도 제한 초과
                speed_color = (255, 0, 0)
            elif current_speed > 50:
                speed_color = (255, 165, 0)
            
            speed_surface = percent_font.render(speed_text, True, speed_color)
            display.blit(speed_surface, (indicator_x + 100, indicator_y + 45))
            
            # 속도 제한 표시
            limit_text = "LIMIT: 60"
            limit_surface = level_font.render(limit_text, True, (128, 128, 128))
            display.blit(limit_surface, (indicator_x + 200, indicator_y + 55))
            
            # === 위험도 바 ===
            bar_x = indicator_x + 250
            bar_y = indicator_y + 15
            bar_width = 120
            bar_height = 15
            
            # 바 배경
            pygame.draw.rect(display, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            
            # 위험도에 따른 바 채우기
            fill_width = int(bar_width * crash_prob)
            if fill_width > 0:
                pygame.draw.rect(display, risk_color, (bar_x, bar_y, fill_width, bar_height))
            
            # 바 테두리
            pygame.draw.rect(display, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)
            
            # === 속도 바 ===
            speed_bar_y = indicator_y + 45
            speed_bar_width = int(bar_width * min(current_speed / 100.0, 1.0))  # 100km/h 기준
            
            # 속도 바 배경
            pygame.draw.rect(display, (30, 30, 30), (bar_x, speed_bar_y, bar_width, bar_height))
            
            # 속도 바 채우기
            if speed_bar_width > 0:
                speed_bar_color = speed_color
                pygame.draw.rect(display, speed_bar_color, (bar_x, speed_bar_y, speed_bar_width, bar_height))
            
            # 속도 바 테두리 (노란선 제거됨)
            pygame.draw.rect(display, (255, 255, 255), (bar_x, speed_bar_y, bar_width, bar_height), 1)
            
        except Exception as e:
            pass  # pygame 관련 오류는 무시

    def render_yolo_detections(self, display):
        """YOLO 감지 결과를 화면에 박스로 표시"""
        try:
            import pygame
            
            if not self.crash_detector.detected_objects:
                return
            
            # 클래스 이름 매핑 (차량만)
            class_names = {
                2: 'car'
            }
            
            # 클래스별 색상 (차량만)
            class_colors = {
                2: (0, 255, 0)    # car - 초록
            }
            
            # 화면 크기
            screen_width = self.dim[0]
            screen_height = self.dim[1]
            
            font = pygame.font.Font(None, 24)
            
            for obj in self.crash_detector.detected_objects:
                if obj['confidence'] > 0.3:  # 신뢰도 30% 이상만 표시
                    # YOLO 좌표를 화면 좌표로 변환 (640x640 -> 실제 화면 크기)
                    x1, y1, x2, y2 = obj['bbox']
                    
                    # 좌표 스케일링
                    x1 = int(x1 * screen_width / 640)
                    y1 = int(y1 * screen_height / 640) 
                    x2 = int(x2 * screen_width / 640)
                    y2 = int(y2 * screen_height / 640)
                    
                    # 화면 경계 내로 제한
                    x1 = max(0, min(x1, screen_width))
                    y1 = max(0, min(y1, screen_height))
                    x2 = max(0, min(x2, screen_width))
                    y2 = max(0, min(y2, screen_height))
                    
                    cls = obj['class']
                    conf = obj['confidence']
                    
                    # 박스 색상
                    color = class_colors.get(cls, (255, 255, 255))
                    
                    # 감지 박스 그리기
                    pygame.draw.rect(display, color, (x1, y1, x2-x1, y2-y1), 2)
                    
                    # 라벨 텍스트
                    label = f"{class_names.get(cls, 'unknown')} {conf:.2f}"
                    text_surface = font.render(label, True, color)
                    
                    # 텍스트 배경
                    text_rect = text_surface.get_rect()
                    text_bg = pygame.Surface((text_rect.width + 4, text_rect.height + 4))
                    text_bg.set_alpha(200)
                    text_bg.fill((0, 0, 0))
                    
                    # 텍스트 위치 (박스 위쪽)
                    text_x = x1
                    text_y = max(0, y1 - text_rect.height - 4)
                    
                    display.blit(text_bg, (text_x, text_y))
                    display.blit(text_surface, (text_x + 2, text_y + 2))
                    
        except Exception as e:
            pass

# 원본 manual_control의 HUD를 우리 확장 버전으로 교체
def patched_game_loop(args):
    """원본 game_loop을 패치하여 우리 HUD 사용"""
    # 원본 game_loop 코드를 복사하되 HUD 부분만 교체
    import pygame
    
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None

    try:
        client = manual_control.carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                # 시뮬레이션 속도에 따른 delta_seconds 조정
                target_fps = args.fps
                delta_seconds = (1.0 / target_fps) / args.sim_speed
                settings.fixed_delta_seconds = delta_seconds
                print(f"🕐 시뮬레이션 설정: {target_fps} FPS, 속도 {args.sim_speed}x, delta={delta_seconds:.3f}s")
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        # 우리의 확장된 HUD 사용
        hud = CrashDetectionHUD(args.width, args.height, args.sim_speed, args.fps)
        world = manual_control.World(sim_world, hud, args)
        
        # 속도 제한이 있는 컨트롤러 사용 (기본 60km/h 제한)
        max_speed = 60  # 최대 속도 설정 (km/h)
        controller = SpeedLimitedKeyboardControl(world, args.autopilot, max_speed)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            if args.sync:
                sim_world.tick()
            # 설정된 FPS 사용
            clock.tick_busy_loop(args.fps)
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()

def main():
    """메인 함수"""
    print("🚗 CARLA Manual Control + Crash Detection")
    print("원본 CARLA manual_control에 crash detection 모델 통합")
    print("="*60)
    
    # 원본 manual_control의 argparser 사용
    argparser = manual_control.argparse.ArgumentParser(
        description='CARLA Manual Control with Crash Detection')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1, example: 192.168.0.100 for remote server)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',  # 테슬라 모델3 승용차로 변경
        help='actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--fps',
        metavar='N',
        default=20,
        type=int,
        help='Set target FPS for simulation (default: 20)')
    argparser.add_argument(
        '--sim-speed',
        metavar='SPEED',
        default=1.0,
        type=float,
        help='Simulation speed multiplier (default: 1.0 = real-time)')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = manual_control.logging.DEBUG if args.debug else manual_control.logging.INFO
    manual_control.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    manual_control.logging.info('listening to server %s:%s', args.host, args.port)

    print(manual_control.__doc__)

    try:
        patched_game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()

