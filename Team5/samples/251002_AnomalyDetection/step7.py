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
# best_obstacle_detection_model.pth : 이전에 훈련 중 검증 손실이 가장 낮았던 최적의 가중치 파일
model.load_state_dict(torch.load('best_obstacle_detection_model.pth', map_location=device))
model.eval()
model = model.to(device)

# 2. 모델을 추적하여 TorchScript로 변환 (Tracing)
# 모델이 요구하는 입력 이미지 형식과 동일한 더미 입력 생성
# torch.randn : PyTorch에서 특정 모양을 가진 텐서를 생성하는 함수
# torch.randn(배치크기, 채널, 높이, 너비)
# 배치크기 : 한 번에 처리할 이미지 개수
# 채널 = 3 : RGB이므로 3개로  설정
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# TorchScript로 변환
# PyTorch 모델을 TorchScript라는 형식으로 변환하는 과정
# Python 환경 밖에서도 독립적으로 실행할 수 있도록 직렬화하고 최적화하는 단계
# 직렬화 : 딥러닝 모델의 데이터와 구조를 영구적으로 저장하고 전송하기 쉬운 형태(파일)로 변환하는 것
# torch.jit.trace : PyTorch 모델을 TorchScript로 변환하는 함수
# TorchScript : PyTorch 모델을 직렬화하고 최적화하여 Python 환경에 의존하지 않고도 실행할 수 있도록 하는 중간표현언어
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
        # 이미지를 불러오고, RGB 형식으로 변환을 함
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {image_path}")
        return False, 0.0
        
    input_tensor = inference_transforms(img).unsqueeze(0) # [1, 3, 224, 224] 텐서로 변환
    # inference_transforms : 1. 이미지 크기를 조정(Resize), 2. 이미지의 형태를 H x W x C(높이, 너비, 채널)에서 C x H x W 형태의 텐서로 변환하고 픽셀 값을 0과 1사이로 정규화 시켜줌. 3. 평균, 표준편차를 통해서 정규화
    # inference_transforms = 1단계에서 했던 전처리를 해주는 과정임
    # .unsqueeze(0) : 텐서의 가장 앞쪽 인덱스에 새로운 차원을 하나 추가, (0)을 통해서 배치 크기 1을 추가해줌
    
    # 모델 추론
    with torch.no_grad():
        output = deployed_model(input_tensor)
        # 가공되지 않은 숫자(Logit)을 얻음
        
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
