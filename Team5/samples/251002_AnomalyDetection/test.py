import cv2
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import numpy as np
import os
import time

# ==========================================================
# 1. 모델 환경 설정 및 로드 (이전 단계에서 정의된 값 사용)
# ==========================================================

DEPLOYMENT_FILE = "deployed_obstacle_detector.pt" # 7단계에서 저장한 모델 파일
OPTIMAL_THRESHOLD = 0.8 # 6단계에서 결정된 최적 임계값
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_INPUT_SIZE = 224

# --- 모델 구조 정의 (로드 시 필요) ---
def build_model(num_classes=1):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 모델 로드 ---
try:
    # TorchScript 모델 로드
    deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
    deployed_model.eval()
    print(f"✅ 배포 모델 '{DEPLOYMENT_FILE}' 로드 완료.")
except Exception as e:
    print(f"❌ 오류: 배포 파일 로드 실패. '{DEPLOYMENT_FILE}' 파일과 경로를 확인하세요. 오류: {e}")
    sys.exit()

# --- 이미지 전처리 파이프라인 ---
inference_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# ==========================================================
# 2. 웹캠 감지 기능 추가 (기존 OpenCV 루프 통합)
# ==========================================================

# 1. VideoCapture 객체 생성 및 인덱스 확인 (이전 안내대로 0, 1, 2 등으로 테스트 필요)
CAMERA_INDEX = 0 # 🚨 검은 화면 시 1, 2 등으로 변경하여 다시 시도하세요.
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌ 오류: 웹캠 (인덱스 {CAMERA_INDEX})을 열 수 없습니다. 인덱스를 확인하거나 다른 프로그램을 종료하세요.")
    sys.exit()
    
print("\n💡 실시간 감지 시작. 'q' 키를 눌러 종료합니다.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read() # 프레임을 읽어옴. (OpenCV BGR 형식)

    if not ret:
        print("프레임을 받지 못하고 종료합니다.")
        break
    
    # --- [기능 추가: 이상/정상 감지] ---
    
    # 1. OpenCV BGR 이미지를 모델이 요구하는 PIL Image (RGB)로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # 2. 이미지 전처리 및 텐서 변환
    input_tensor = inference_transforms(pil_img).unsqueeze(0)
    
    # 3. 모델 추론
    with torch.no_grad():
        output = deployed_model(input_tensor)
        
    prob = torch.sigmoid(output).item() # 확률 (0~1)
    is_obstacle = prob >= OPTIMAL_THRESHOLD

    # 4. 결과 표시 텍스트 및 색상 결정
    if is_obstacle:
        status = "ANOMALY: OBSTACLE DETECTED"
        color = (0, 0, 255) # 빨간색
    else:
        status = "NORMAL"
        color = (0, 255, 0) # 녹색

    # 5. FPS 계산 (선택 사항)
    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    
    # 6. 윈도우 창에 결과 텍스트 오버레이
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Prob: {prob:.4f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


    # 윈도우 창에 프레임 표시
    cv2.imshow('Webcam Live - Anomaly Detector', frame)

    # 1ms 동안 키 입력 대기. 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 사용이 끝난 후 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
print("\n감지 프로그램 종료.")