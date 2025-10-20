import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# from sklearn.metrics import ... # 평가 지표는 5단계에서 필요

# ==========================================================
# 1단계: 데이터 전처리 및 탐색
# ==========================================================

# 1-1. 기본 설정 (필수 수정)
# ----------------------------------------------------------
# 데이터셋의 루트 경로를 설정하세요. (필수 수정)
DATA_ROOT = '/home/ubuntu26/workspace/AD_Dataset/' # 데이터셋의 기본 경로를 설정 해줌
NORMAL_DIR = os.path.join(DATA_ROOT, 'normal_frames') # 기본 경로에서 정상 데이터셋으로 사용할 경로를 설정
ANOMALY_DIR = os.path.join(DATA_ROOT, 'anomaly_frame') # 기본 경로에서 이상 데이터셋으로 사용할 경로를 설정

IMAGE_SIZE = 224 # 사용되는 딥러닝 모델에 이미지를 입력하기 위해서 모든 이미지의 크기를 통일시키기 위해서 픽셀값을 설정
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 1-2. 이미지 전처리 정의
# ----------------------------------------------------------
train_transforms = transforms.Compose([ # 모델이 학습할 데이터를 적용, 데이터 증강 기법을 추가하여 모델의 일반화 능력을 높이고 과적합을 방지해주는 역할
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 딥러닝 모델에 맞게 이미지 픽셀 크기를 수정해줌
    transforms.RandomHorizontalFlip(),  # 이미지를 무작위로 좌우 반전(데이터 증강 : 모델이 좌우가 바뀐 장애물도 인식하도록 학습하여 다양성을 확보)
    transforms.RandomRotation(10),  # 이미지를 -10도~10도 무작위로 회전(장애물이 약간 기울어진 상태에서도 인식하도록 강인성을 높임)
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서 형식으로 변환, 픽셀값을 0.0~0.1로 정규화
    transforms.Normalize(MEAN, STD) # 정규화를 통해서 학습 안정성, 속도를 높이고, 전이 학습 효과를 극대화 시킴
])

val_test_transforms = transforms.Compose([  # 모델 성능을 평가할 데이터에 적용, 모델의 성능을 왜곡할 수 있는 증강이 절대 포함되지 않으며, 데이터형식을 통일하고 안정화하는 필수 전처리만 수행
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 1-3. 커스텀 데이터셋 클래스 생성
# ----------------------------------------------------------
class ShipObstacleDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.all_files = files # 파일 목록 : 훈련에 사용할 모든 이미지 파일 경로를 저장
        self.all_labels = labels # 라벨 목록 : 사진에 정상(0), 이상(1)을 파악하여 이름에 붙이고 저장
        self.transform = transform # 전처리 : 이미지에 적용할 전처리 파이프라인
        
    def __len__(self):
        return len(self.all_files) # 데이터셋의 이미지 개수를 반환

    def __getitem__(self, idx):
        img_path = self.all_files[idx] # 이미지 경로를 가져옴
        label = self.all_labels[idx]   # 라벨링된 이름의 경로를 가져옴
        image = Image.open(img_path).convert('RGB') # 사용할 이미지를 불러오고, 모든 이미지를 RGB 3채널 형식으로 통일
        
        if self.transform:
            image = self.transform(image) # 이미지를 텐서로 변환
            
        label = torch.tensor(label, dtype=torch.float32) # 정수형으로 저장된 정상,이상 판단하는 인덱스를 32비트 실수형 텐서로 변환(GPU가 효율적으로 처리하기 위함)
        return image, label # 변환된 이미지, 라벨을 반환

# 1-4. 전체 데이터셋 파일 리스트 및 레이블 초기 생성 (탐색)
# ----------------------------------------------------------
normal_files = glob.glob(os.path.join(NORMAL_DIR, '*.jpg')) + glob.glob(os.path.join(NORMAL_DIR, '*.png'))  # 정상이미지있는 경로와 jpg(png)를 결합하여 리스트로 반환
anomaly_files = glob.glob(os.path.join(ANOMALY_DIR, '*.jpg')) + glob.glob(os.path.join(ANOMALY_DIR, '*.png'))   # 이상이미지있는 경로와 jpg(png)를 결합하여 리스트로 반환
files = normal_files + anomaly_files # 두 파일을 합쳐서 전체 이미지 경로를 만듦
labels = [0] * len(normal_files) + [1] * len(anomaly_files) # 정상, 이상 각 개수만큼 [0], [1]을 반복

print(f"총 이미지 수: {len(files)}, 정상: {len(normal_files)}, 이상: {len(anomaly_files)}")


# ==========================================================
# 2단계: 데이터 분할 및 불균형 처리
# ==========================================================

# 2-1. 데이터 분할 (Stratified Split으로 불균형 비율 유지)
# ----------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
# files, labels(전체 이미지와 라벨링 경로)
# test_size=0.2 : 전체 데이터의 20%를 임시 세트(X_temp, y_temp)로 분리
# random_state=42 : 데이터를 무작위로 분리할 때 일관성을 유지하기 위한 시드 값, 이 값이 같으면 언제 실행해도 동일하게 분리(42는 임의의 숫자, 바꿔도 상관없지만 고정되어야 함)
# stratify=labels : 정상 또는 이상 클래스가 어느 한 쪽 세트에 몰리는 것을 방지하여, 모델이 학습 및 테스트 단계에서 모든 유형의 데이터를 고르게 보도록 보장 = 공정한 성능 평가를 위함
 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# X_temp, y_temp : 1차 분할에서 남은 임시 세트(전체 20%)를 입력으로 사용
# test_size=0.5 : 임시 세트의 50%를 테스트 세트로 분리 = 전체 10%
# X_val, y_val : 나머지 50%(10%)는 검증 세트로 사용
# X_test, y_test : 최종 테스트 세트. 최종 성능 평가
# stratify=y_temp : 1차와 같이 검증 세트와 테스트 세트를 균등하게 분배하도록 함

print(f"학습/검증/테스트 데이터 수: {len(X_train)} / {len(X_val)} / {len(X_test)}")

# 2-2. 클래스 가중치 계산 (불균형 처리 핵심)
# ----------------------------------------------------------
# 훈련 데이터셋 안에서 이상, 정상 데이터의 정확한 개수 파악
num_train_anomaly = sum(y_train)    # 이상 데이터의 개수(1+0...)
num_train_normal = len(y_train) - num_train_anomaly # 정상 데이터의 개수
total_train_samples = len(y_train)  # 전체 데이터의 개수

# 클래스 가중치 계산 공식 적용
# 데이터 불균형을 해결하기 위해서 클래스 가중치를 계산하고, 이를 딥러닝 모델의 손실 함수에 적용하기 위해 준비하는 과정
weight_normal = total_train_samples / (2 * num_train_normal)
weight_anomaly = total_train_samples / (2 * num_train_anomaly)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pos_weight = torch.tensor([weight_anomaly / weight_normal], dtype=torch.float32).to(device) # BCEWithLogitsLoss에 필요한 긍정 클래스 가중치
# 긍정 클래스 가중치 : 이진 분류 문제에서 소수 클래스, 즉 긍정(1) 클래스의 중요도를 높여 모델 학습에 균형을 맞추기 위해서 사용되는 값
# 정상(0) 데이터셋의 개수가 7000장이고, 이상(1) 데이터셋의 개수가 2~3000장으로 많이 적기 때문에 이로 인해 발생하는 불균형을 해소하기 위해서 사용
# 긍정 클래스 가중치를 사용하면 모델은 틀리면 안 되는 중요한 데이터(이상)에 학습 자원을 집중하게 되어, 결과적으로 위음성(FN)을 줄이는 데 효과적

print(f"클래스 가중치 비율 (이상/정상): {pos_weight.item():.2f}")


# ==========================================================
# 3단계: 모델 선택 및 구축
# ==========================================================

# 딥러닝 모델의 구조를 정의하고 초기화
# 사전 훈련된 ResNet18을 불러와 이진 분류에 사용하기 위해서 수정하는 부분이라고 보면 됨

def build_model(num_classes=1, pretrained=True):
    # pretrained=True : 모델의 가중치를 ImageNet이라는 대규모 데이터셋으로 미리 학습된 값으로 초기화(전이학습)
    # ResNet-18 사용 (사전 학습된 가중치로 빠른 성능 확보)
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    # 이진 분류를 위해 최종 레이어를 1개 출력으로 수정
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

model = build_model(num_classes=1, pretrained=True)
model = model.to(device)


# ==========================================================
# 4단계: 모델 학습 및 최적화
# ==========================================================

# 4-1. 학습 환경 및 하이퍼파라미터 설정
# ----------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20 # 20 에포크로 설정
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 최적화 도구는 손실(오차)를 최소화하기 위해 모델의 가중치를 조정하는 알고리즘

# 불균형 가중치가 적용된 손실 함수
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# criterion : 손실 함수는 모델의 예측 값과 데이터셋의 정답 레이블 사이의 오차를 계산. 모델은 이 오차를 최소화하도록 함
# 

# 4-2. 데이터로더 준비
# ----------------------------------------------------------
# 앞서 8:1:1로 나누었던 데이터들을 사용하여 3가지 객체 생성
train_dataset = ShipObstacleDataset(X_train, y_train, train_transforms)
val_dataset = ShipObstacleDataset(X_val, y_val, val_test_transforms)
test_dataset = ShipObstacleDataset(X_test, y_test, val_test_transforms)

# DataLoader 객체는 딥러닝 모델 학습을 위 데이터를 가장 효율적이고 체계적으로 공급하는 핵심 도구
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # 훈련할 때는 섞어서
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)   # 검증할 때는 재현성을 위해 섞지 않고
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) # 5단계에서 사용

# 4-3. 학습 루프 구현 및 실행
# ----------------------------------------------------------
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device):
    best_val_loss = float('inf')    # 최저 검증 손실 초기화
    
    for epoch in range(num_epochs): # 에포크만큼반복 
        model.train()   # 모델을 훈련 모드로 설정
        running_loss = 0.0  # 현재 에포그의 누적 손실 초기화 : 모델이 전체 훈련 데이터셋을 한 번 처리하는 동안 발생하는 누적된 오차의 총량
        
        # train_loader에서 배치(inputs, labels)를 하나씩 가져와 모델을 학습시키는 부분
        for inputs, labels in train_loader: 
            inputs, labels = inputs.to(device), labels.to(device) # GPU에서 사용하기 위하서 이동시킴

            optimizer.zero_grad() # 기울기 초기화
            outputs = model(inputs).squeeze(1)  # 모델에 입력을 넣어줘서 예측값을 얻음, squeeze(1)은 출력 형태를 정답 레이블과 일치시키기 위함
            loss = criterion(outputs, labels)   # 손실 함수를 사용해서 오차를 구함
            
            loss.backward() # 역전파 : 계산된 오차를 바탕으로 모델 가중치들의 기울기를 계산
            optimizer.step() # 가중치 업데이트 : 계산된 기울기를 사용하여 Adam 모델의 가중치를 실제로 수정
            
            running_loss += loss.item() * inputs.size(0)    # 현재 배치 손실을 전체 훈련 손실에 누적시킴. loss.item(): 배치 당 평균 손실. inputs.size(0) : 배치 크기
            
        train_loss = running_loss / len(train_loader.dataset)   # 누적된 전체 손실을 전체 훈련 샘플 개수로 나누어 해당 에포크의 평균 훈련 손실을 계산

        # 설정한 에포크만큼 반복을 해줌. 훈련을 통해 에포크 당 훈련 손실을 계산해서 최저 검증 손실을 업데이트
        
        # 검증 단계
        model.eval()    # 검증 모드로 전환
        running_val_loss = 0.0  # 검증에서 사용될 현재 에포크의 누적 손실 초기화
        with torch.no_grad():   # 이 블록 내부에서는 기울기 계산을 일시적으로 비활성화를 시켜줌
                                # 검증 단계에서는 모델의 가중치를 업데이트할 필요가 없기 때문(학습 X), 기울기 계산에 필요한 메모리와 연산 시간을 절약하여 검증 속도를 높여줌
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        val_loss = running_val_loss / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_obstacle_detection_model.pth')
            print(" -> 최적 모델 저장 완료!")
            
    print('학습 완료!')

    # 에포크만큼 학습하면서 가장 잘 동작했던 시험의 모델 상태를 복원하여, 최종적으로 가장 우수한 성능을 가진 모델을 사용자에게 제공
    model.load_state_dict(torch.load('best_obstacle_detection_model.pth'))
    return model

# --- 학습 실행: 이 줄의 주석을 해제하면 학습이 시작됩니다. ---
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, device)

# 준비하고 정의했던 모든 구성요소를 train_model 함수에 전달하고, 함수가 반환하는 최종결과를 trained_model에 저장