#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pickle
from collections import deque
import cv2
import base64
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
import math
import sys
import importlib.util as _importlib_util

# YOLO 모델 로드를 위한 경로 추가
# 이 스크립트가 'pred_danger' 폴더 안에 있다고 가정합니다.
# 'yolov7' 폴더가 'pred_danger' 폴더와 같은 레벨에 있어야 합니다.
# 예: v2x_server/yolov7, v2x_server/pred_danger
yolo_path = os.path.join(os.path.dirname(__file__), '..', 'yolov7')
if os.path.exists(yolo_path):
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    print(f"✅ Added to sys.path: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
else:
    # Fallback if yolov7 is inside pred_danger
    yolo_path_alt = os.path.join(os.path.dirname(__file__), 'yolov7')
    if os.path.exists(yolo_path_alt):
        sys.path.append(os.path.dirname(__file__))
        print(f"✅ Added to sys.path: {os.path.abspath(os.path.dirname(__file__))}")
    else:
        print(f"⚠️ YOLOv7 directory not found at {yolo_path} or {yolo_path_alt}. Object detection will be disabled.")


# --- 모델 및 유틸리티 클래스 정의 ---

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

class YOLODetector:
    """YOLO 객체 감지 모듈"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.img_size = 640
        self.class_names = []
        self.load_yolo_model()
        print(f"YOLODetector initialized on device: {self.device}")

    def load_yolo_model(self):
        """YOLO 모델 로드"""
        try:
            # 'yolov7' 폴더가 상위 디렉토리에 있다고 가정
            model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov7', 'yolov7.pt')
            if not os.path.exists(model_path):
                 # 'yolov7' 폴더가 현재 디렉토리에 있다고 가정
                model_path = os.path.join(os.path.dirname(__file__), 'yolov7', 'yolov7.pt')

            if os.path.exists(model_path):
                from models.experimental import attempt_load
                self.model = attempt_load(model_path, map_location=self.device)
                self.model.eval()
                self.class_names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
                print("✅ YOLOv7 model loaded successfully.")
            else:
                print(f"⚠️ YOLOv7 model file not found at {model_path}. Detection will be disabled.")
                self.model = None
        except Exception as e:
            print(f"❌ Failed to load YOLOv7 model: {e}. Detection will be disabled.")
            self.model = None

    def detect_objects(self, image):
        """객체 감지 실행"""
        if self.model is None:
            return []

        try:
            from utils.general import non_max_suppression, scale_coords
            from utils.datasets import letterbox

            # 이미지 전처리
            img0 = image.copy()
            img = letterbox(img0, self.img_size, stride=self.model.stride.max())[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 추론
            with torch.no_grad():
                pred = self.model(img)[0]
                pred = non_max_suppression(pred, 0.25, 0.45, classes=[2, 5, 7], agnostic=False) # car, bus, truck

            detections = []
            if pred is not None and len(pred):
                for det in reversed(pred):
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            x1, y1, x2, y2 = map(int, xyxy)
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'confidence': conf.item(),
                                'class': int(cls),
                                'label': self.class_names[int(cls)]
                            })
            return detections
        except Exception as e:
            print(f"❌ Error during YOLO detection: {e}")
            return []


class CrashDetectionModuleForServer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 60
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        self.yolo_detector = YOLODetector()
        self.scaler = None
        self.trained_model = None
        self.load_scaler()
        self.load_trained_model()
        print(f"CrashDetectionModule (Server) initialized on device: {self.device}")

    def load_trained_model(self):
        try:
            # 모델 경로 수정 (pred_danger/model/...)
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model_front_classification.pth')
            model_config = {
                'c_in': 16, 'seq_len': 60, 'patch_len': 16, 'stride': 8, 'd_model': 128,
                'n_heads': 8, 'n_layers': 3, 'd_ff': 256, 'dropout': 0.1, 'n_classes': 2
            }
            if os.path.exists(model_path):
                model = PatchTSTCollisionPredictor(**model_config).to(self.device)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                self.trained_model = model
                print("✅ Trained PatchTST model loaded.")
            else:
                self.trained_model = None
                print(f"⚠️ Trained model file not found at {model_path}")
        except Exception as e:
            self.trained_model = None
            print(f"❌ Failed to load trained model: {e}")

    def load_scaler(self):
        # 스케일러 경로 수정
        scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'scaler_front_classification.pkl')
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✅ Scaler loaded from: {scaler_path}")
            else:
                self.scaler = None
                print(f"⚠️ Scaler file not found at {scaler_path}")
        except Exception as e:
            self.scaler = None
            print(f"❌ Failed to load scaler: {e}")

    def _calculate_features(self, ego_data, detected_objects, image_shape):
        """
        YOLO 결과와 ego 차량 데이터로 특징 벡터 생성
        """
        # ego 차량 정보
        ego_speed = (ego_data['velocity_x']**2 + ego_data['velocity_y']**2 + ego_data['velocity_z']**2)**0.5
        ego_vx, ego_vy = ego_data['velocity_x'], ego_data['velocity_y']
        
        # TODO: API로 ego 차량의 위치(x,y)와 yaw 값을 받아야 더 정확한 계산이 가능합니다.
        # 현재는 없으므로 상대 위치 계산을 할 수 없습니다.
        # 임시로 이미지 중앙 하단을 ego 차량 위치로 가정합니다.
        h, w, _ = image_shape
        ego_x, ego_y = w / 2, h 
        
        # 가장 위험한 객체(가장 가까운 객체)의 특징을 사용
        min_dist_sq = float('inf')
        best_feature_vector = [0.0] * 16 # 특징 벡터 초기화

        if not detected_objects:
            best_feature_vector[5] = ego_speed # ego_speed
            return best_feature_vector

        for obj in detected_objects:
            box = obj['box']
            # 객체 위치를 박스 하단 중앙으로 추정
            obj_x, obj_y = (box[0] + box[2]) / 2, box[3]
            
            # 상대 위치 (카메라 좌표계)
            relative_x = obj_x - ego_x
            relative_y = obj_y - ego_y # 이미지에서는 y가 아래로 증가하므로 음수값
            
            dist_sq = relative_x**2 + relative_y**2
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                
                # TODO: 상대 속도, TTC 등은 3D 월드 좌표계 정보 없이는 정확한 계산이 어렵습니다.
                # 여기서는 단순화된 추정을 사용합니다.
                # 예: 박스 크기 변화율로 접근 속도 추정 (연속 프레임 필요)
                # 이 예제에서는 속도 관련 특징은 ego 속도만 사용하고 나머지는 0으로 둡니다.
                relative_vx = 0.0 
                relative_vy = 0.0
                relative_speed = 0.0
                ttc = 99.0 # Time to collision
                
                feature_vector = [0.0] * 16
                feature_vector[0] = relative_x
                feature_vector[1] = relative_y
                feature_vector[2] = relative_vx
                feature_vector[3] = relative_vy
                feature_vector[4] = relative_speed
                feature_vector[5] = ego_speed
                # ... 기타 특징들 (가속도, 거리 등)
                feature_vector[9] = math.sqrt(dist_sq) # ego_distance
                feature_vector[12] = ttc
                
                best_feature_vector = feature_vector

        return best_feature_vector

    def process_and_predict(self, image, ego_data):
        """
        이미지 처리, 특징 추출, 충돌 예측을 모두 수행
        """
        # 1. YOLO로 객체 탐지
        detected_objects = self.yolo_detector.detect_objects(image)

        # 2. 특징 벡터 계산
        feature_vector = self._calculate_features(ego_data, detected_objects, image.shape)
        
        # 3. 버퍼에 특징 벡터 추가
        self.feature_buffer.append(feature_vector)

        # 4. 모델로 예측 (버퍼가 충분히 찼을 경우)
        if self.trained_model and len(self.feature_buffer) >= self.sequence_length:
            try:
                sequence_data = np.array(list(self.feature_buffer))
                if self.scaler:
                    sequence_data = self.scaler.transform(sequence_data)
                
                input_tensor = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.trained_model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=-1)
                    return probabilities[0][1].item(), detected_objects
            except Exception as e:
                print(f"❌ Model prediction error: {e}")
                return 0.1, detected_objects
        else:
            # 버퍼가 채워질 때까지 휴리스틱 값 또는 0을 반환
            return 0.0, detected_objects

# FastAPI 앱 초기화
app = FastAPI()

# 추론 모듈 인스턴스 생성
detector = CrashDetectionModuleForServer()

# API 요청 본문을 위한 데이터 모델 정의
class InferenceRequest(BaseModel):
    # 이미지는 Base64 인코딩된 문자열로 받음
    image_base64: str
    ego_velocity_x: float
    ego_velocity_y: float
    ego_velocity_z: float
    # 필요에 따라 더 많은 ego 차량 데이터를 추가할 수 있습니다.

@app.post("/predict")
def predict_crash(request: InferenceRequest):
    # Base64 문자열을 디코딩하여 이미지로 변환
    try:
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_image is None:
            raise ValueError("Failed to decode image, imdecode returned None.")
    except Exception as e:
        return {"error": f"Image decoding failed: {e}"}

    # ego 차량 데이터 정리
    ego_data = {
        "velocity_x": request.ego_velocity_x,
        "velocity_y": request.ego_velocity_y,
        "velocity_z": request.ego_velocity_z,
    }

    # 모델로 예측
    probability, detections = detector.process_and_predict(cv_image, ego_data)
    
    return {
        "crash_probability": probability,
        "detected_objects_count": len(detections),
        "detections": detections
    }

if __name__ == "__main__":
    # 서버 실행 (uvicorn inference_server:app --host 0.0.0.0 --port 8000)
    print("Starting FastAPI server...")
    print("Run with: uvicorn inference_server:app --host 0.0.0.0 --port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)