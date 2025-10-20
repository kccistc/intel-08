import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split # 데이터셋 분할을 위해 필요

# ==========================================================
# 1. 초기 환경 설정 및 모델/데이터셋 구조 재정의 (1~4단계와 동일)
# ==========================================================

# 🚨 학습 환경과 동일하게 설정해야 합니다.
DATA_ROOT = '/home/ubuntu26/workspace/AD_Dataset/' 
NORMAL_DIR = os.path.join(DATA_ROOT, 'normal_frames')
ANOMALY_DIR = os.path.join(DATA_ROOT, 'anomaly_frame')
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# --- 3단계. 모델 구조 정의 ---
def build_model(num_classes=1):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 7-1. 모델 경량화 및 변환 ---

# 1. 최적 가중치 로드
# 배포 환경은 일반적으로 CPU를 사용하므로 device를 'cpu'로 설정합니다.
device = torch.device("cpu") 
model = build_model(num_classes=1)
# 🚨 best_obstacle_detection_model.pth 파일이 현재 디렉토리에 있어야 합니다.
model.load_state_dict(torch.load('best_obstacle_detection_model.pth', map_location=device))
model.eval()
model = model.to(device)

# 2. 모델을 추적하여 TorchScript로 변환 (Tracing)
# 모델이 요구하는 입력 이미지 형식과 동일한 더미 입력 생성
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# TorchScript로 변환
traced_script_module = torch.jit.trace(model, dummy_input)

# 3. 배포용 파일로 저장
DEPLOYMENT_FILE = "deployed_obstacle_detector.pt"
traced_script_module.save(DEPLOYMENT_FILE)

print("==========================================================")
print("배포용 모델 변환 완료! 파일:", DEPLOYMENT_FILE)
print("==========================================================")


# ==========================================================
# 7-2. 실제 추론(Inference) 코드 예시
# ==========================================================

# 1. 모델 및 임계값 로드 (변환된 파일 사용)
# TorchScript로 저장된 모델을 로드합니다.
deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
OPTIMAL_THRESHOLD = 0.9937 # 6단계에서 결정된 최종 최적 임계값

# 2. 이미지 전처리 정의 (평가 시 사용한 val_test_transforms와 동일해야 함)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def predict_obstacle(image_path):
    """지정된 경로의 이미지에 대해 장애물 감지 추론을 수행합니다."""
    # 이미지 로드 및 전처리
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {image_path}")
        return False, 0.0
        
    input_tensor = inference_transforms(img).unsqueeze(0) # [1, 3, 224, 224] 텐서로 변환
    
    # 모델 추론
    with torch.no_grad():
        output = deployed_model(input_tensor)
        
    # Logits -> Sigmoid -> 확률 (0~1) 변환
    prob = torch.sigmoid(output).item() 
    
    # 임계값 기반 최종 결정
    is_obstacle = prob >= OPTIMAL_THRESHOLD
    
    # 결과 출력
    if is_obstacle:
        status = "🚨 장애물 감지 (Anomaly)"
    else:
        status = "✅ 정상 (Normal)"
        
    print(f"\n--- 추론 결과 (이미지: {os.path.basename(image_path)}) ---")
    print(f"상태: {status}")
    print(f"확률: {prob:.4f} (임계값: {OPTIMAL_THRESHOLD:.4f})")
    
    return is_obstacle, prob

# ----------------------------------------------------------------------
# 🔥 실행 예시 🔥
# ----------------------------------------------------------------------
# NOTE: 이 예시는 추론 테스트를 위한 것입니다. 실제 이미지 경로로 대체하세요.
# 만약 6단계에서 발견된 FN 이미지를 다시 테스트하고 싶다면 해당 경로를 사용하세요.

# 예시 1: 6단계에서 발견된 FN 이미지 (실제 이상이지만 모델이 놓친 이미지)
FN_IMAGE_PATH = os.path.join(ANOMALY_DIR, 'yt077_00_00014.jpg')
print("\n[테스트 1: FN 이미지 재검증]")
predict_obstacle(FN_IMAGE_PATH)


# 예시 2: 정상 이미지 (테스트 데이터셋 중 하나를 임의로 지정하여 테스트)
# 🚨 이 부분은 실제 정상 이미지 경로로 수정해야 합니다. 
# 만약 정상 파일 경로를 모른다면, 이 줄은 주석 처리하세요.
# NORMAL_TEST_PATH = os.path.join(NORMAL_DIR, '임의의_정상_이미지_이름.jpg') 
# print("\n[테스트 2: 정상 이미지 검증]")
# predict_obstacle(NORMAL_TEST_PATH)