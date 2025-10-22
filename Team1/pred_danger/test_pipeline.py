import torch
import cv2
import numpy as np
import pickle
from collections import deque
from pathlib import Path
import torch.nn as nn
import math
import time # ✨ 시간 측정을 위해 time 모듈 추가

# ==========================================================
# ✨ 1. 설정: 파일 경로를 다시 한번 확인해주세요 ✨
# ==========================================================
# YOLOv7 코드가 있는 폴더 경로
YOLO_REPO_PATH = './yolov7'
# 사전훈련된 YOLOv7-tiny 모델 가중치(.pt) 파일 경로
YOLO_MODEL_PATH = './yolov7/yolov7-tiny.pt'  # 사전훈련된 tiny 모델 사용

# PatchTST 분류 모델 경로
PATCHTST_MODEL_PATH = './model/best_model_front_classification_ex.pth'
SCALER_PATH = './model/scaler_front_classification_ex.pkl'

# 테스트할 비디오 파일 경로
VIDEO_SOURCE = './test/test.mp4'

# 모델 하이퍼파라미터 (학습 때와 동일하게)
SEQUENCE_LENGTH = 60
N_FEATURES = 16

# ==========================================================
# ✨ 2. 분류(Classification) 모델 클래스 정의 ✨
# ==========================================================
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
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(d_model * n_patches, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, x):
        x = self.patch_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x + self.pos_encoding)
        x = self.encoder(x)
        return self.classifier(x)

# ==========================================================
# ✨ 3. 메인 테스트 함수 (분류 모델에 맞게 수정) ✨
# ==========================================================
def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading models...")
    yolo_model = torch.hub.load(YOLO_REPO_PATH, 'custom', YOLO_MODEL_PATH, source='local')
    
    patchtst_model = PatchTSTCollisionPredictor(c_in=N_FEATURES, seq_len=SEQUENCE_LENGTH)
    patchtst_model.load_state_dict(torch.load(PATCHTST_MODEL_PATH, map_location=torch.device(device)))
    patchtst_model.to(device)
    patchtst_model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    cap = cv2.VideoCapture(str(VIDEO_SOURCE))
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_SOURCE}"); return

    # ==========================================================
    # ✨✨✨ 영상 저장 및 속도 제어 로직 (수정) ✨✨✨
    # ==========================================================
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps) if fps > 0 else 33
    print(f"Original video FPS: {fps:.2f}, Target frame delay: {delay}ms")

    # ✨ 출력 영상 설정
    output_dir = Path('./test')
    output_dir.mkdir(parents=True, exist_ok=True) # ./test 폴더가 없으면 생성
    input_filename = Path(VIDEO_SOURCE).stem # 원본 파일 이름 (확장자 제외)
    # ✨✨✨ 수정된 부분: 출력 파일 형식을 .webm으로 변경 ✨✨✨
    output_path = output_dir / f"output_{input_filename}.webm"
    
    # ✨✨✨ 수정된 부분: .webm 형식으로 저장하기 위한 코덱 설정 (VP8) ✨✨✨
    fourcc = cv2.VideoWriter_fourcc(*'VP80') 
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {output_path}")
    # ==========================================================

    features_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prev_box_info = {}
    current_collision_prob = 0.0

    frame_counter = 0
    SKIP_FRAMES = 3
    last_detections = []

    print("Starting inference... Press 'q' to quit.")
    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret: break

        frame_counter += 1
        main_detection_box = None

        if frame_counter % SKIP_FRAMES == 0:
            results = yolo_model(frame)
            detections = results.xyxy[0].cpu().numpy()
            last_detections = detections
        else:
            detections = last_detections

        current_features = np.zeros(N_FEATURES)
        main_detection_box = None
        if len(detections) > 0:
            best_detection = detections[np.argmax(detections[:, 4])]
            x1, y1, x2, y2, _, _ = best_detection
            main_detection_box = (int(x1), int(y1), int(x2), int(y2))

            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            if prev_box_info:
                delta_x = box_center_x - prev_box_info.get('x', box_center_x)
                delta_y = box_center_y - prev_box_info.get('y', box_center_y)
                current_features[0] = box_center_x / frame.shape[1]
                current_features[1] = box_center_y / frame.shape[0]
                current_features[2] = delta_x
                current_features[3] = delta_y
                current_features[4] = np.sqrt(delta_x**2 + delta_y**2)

            prev_box_info = {'x': box_center_x, 'y': box_center_y}
        
        features_buffer.append(current_features)

        if len(features_buffer) == SEQUENCE_LENGTH:
            sequence_np = np.array(features_buffer)
            sequence_scaled = scaler.transform(sequence_np)
            sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)

            with torch.no_grad():
                output = patchtst_model(sequence_tensor)
                probs = torch.softmax(output, dim=1)
                current_collision_prob = probs[0, 1].item()
        else:
            current_collision_prob = 0.0

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            if (x1, y1, x2, y2) == main_detection_box:
                threshold = 0.2
                color = (0, 0, 255) if current_collision_prob > threshold else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                prob_text = f"Collision: {current_collision_prob:.2%}"
                text_size = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1 - 10), color, -1)
                cv2.putText(frame, prob_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imshow('YOLO + PatchTST Inference', frame)
        
        # ✨ 처리된 프레임을 영상 파일에 저장
        writer.write(frame)
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        wait_time = max(1, delay - int(processing_time_ms))
        
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    # ✨ 루프 종료 후 리소스 해제
    print("Inference finished. Releasing resources...")
    cap.release()
    writer.release() # ✨ VideoWriter 객체 해제
    cv2.destroyAllWindows()
    print(f"Video saved successfully to {output_path}")


if __name__ == '__main__':
    run_inference()

