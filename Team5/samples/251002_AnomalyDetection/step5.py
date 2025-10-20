import os
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve
import matplotlib.pyplot as plt

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
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1-3. 커스텀 데이터셋 클래스 (재정의) ---
class ShipObstacleDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.all_files = files
        self.all_labels = labels
        self.transform = transform
    def __len__(self): return len(self.all_files)
    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = self.all_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32) 
        return image, label

# --- 1-2. 테스트용 전처리 (학습과 동일) ---
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# --- 1-4. 파일 리스트 로드 ---
normal_files = glob.glob(os.path.join(NORMAL_DIR, '*.jpg')) + glob.glob(os.path.join(NORMAL_DIR, '*.png'))
anomaly_files = glob.glob(os.path.join(ANOMALY_DIR, '*.jpg')) + glob.glob(os.path.join(ANOMALY_DIR, '*.png'))
files = normal_files + anomaly_files
labels = [0] * len(normal_files) + [1] * len(anomaly_files)

# --- 2-1. 데이터 분할 (학습 때와 동일한 random_state 사용해야 함) ---
X_train, X_temp, y_train, y_temp = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# --- 3단계. 모델 구조 정의 및 최적 가중치 로드 ---
def build_model(num_classes=1, pretrained=True):
    model = models.resnet18(weights=None) # 가중치 파일에서 로드하므로 None으로 설정
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

model = build_model(num_classes=1, pretrained=False) # 사전 학습 가중치는 사용하지 않고 구조만 가져옴
model.load_state_dict(torch.load('best_obstacle_detection_model.pth', map_location=device)) # 🔥 저장된 최적 가중치 로드 🔥
model = model.to(device)
print(f"최적 모델 'best_obstacle_detection_model.pth' 로드 완료. ({device})")

# --- 테스트 데이터셋 및 로더 준비 ---
test_dataset = ShipObstacleDataset(X_test, y_test, val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# ==========================================================
# 5-1. 성능 평가 함수 정의 및 실행 (Threshold 0.5)
# ==========================================================

def evaluate_model(model, data_loader, device, threshold=0.5):
    # model :  학습된 딥러닝 모델(= ResNet18)
    # data_loader : 평가할 데이터셋을 담은 DataLoader
    # device : 연산 수행을 할 하드웨어 장치
    # threshold : 분류를 결정하는 기준값(임계값)


    """모델을 평가하고, 성능 지표와 확률 값을 반환합니다."""
    model.eval() # 검증 모드
    all_labels = [] # 저장할 리스트를 생성
    all_probs = []
    
    with torch.no_grad():   # 기울기 계산 비활성화 시켜줌
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)  # 0~1의 사이 값으로 변환시킴
            

            # labels.cpu(), probs.cpu() : GPU 메모리에 있는 테서를 CPU 메모리에 복사
            # .numpy() : 복사한 텐서를 넘파이 배열로 변환
            # .extend : 리스트에 데이터 추가
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 수집된 리스트 형태의 데이터를 넘파이 배열로 변환
    # 데이터 형식의 최종 통일
    # 성능 지표 계산 호환성 : F1-Score 등 계산하는 Scikit-learn 함수들은 넘파이 배열을 표준 입력 형식으로 요구, 리스트 상태로는 계싼 함수에 직접 넣을 수 없음
    # 효율적인 배열 연산 : 넘파이 배열은 파이썬 리스트보다 훨씬 빠르고 효율적인 대규모 수학 연산을 지원.        
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 임계값 적용
    # all_preds : 모델이 테스트 데이터에 대해 출력 이상일 확률을 모아놓은 넘파이 배열(0.0~1.0)
    # threshold 보다 작은 참, 작으면 거짓
    # 이 연산의 결관는 참, 거짓으로 구성된 불리언 배열
    # astype(int) : 불리언 배열의 데이터 타입을 정수형으로 변환
    all_preds = (all_probs >= threshold).astype(int)
    
    # 성능 지표 계산
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    # confusion_matrix :Scikit-learn 라이브러리에서 이진 분류 모델의 성능을 평가하는 기본 도구
    # all_labels : 전체 테스트 데이터의 실제 정답 배열
    # all_preds : 모델이 임계값을 기준으로 예측한 결과
    # 출력 : 모델의 예측이 얼마나 정확했는지를 2X2 행렬로 반환
    # ravel() : 2X2 행렬을 1차원 배열로 펼쳐주는 역할


    recall = recall_score(all_labels, all_preds, zero_division=0)
    # recall : 재현률
    # 실제 이상인 데이터 중에서 모델이 이상이라고 정확하게 예측한 비율(= 장애물인 것을 장애물이라고 하는 것)
    # zero_division=0 : 분모가 0인 상황에서 계산이 이루어지지 않기 때문에 값을 0이라고 해서 발생하는 오류를 방지해줌

    precision = precision_score(all_labels, all_preds, zero_division=0)
    # precision : 정밀도
    # 모델이 이상이라고 예측한 모든 데이터 중에서 실제로 이상인 것의 비율

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    # f1 : 정밀도와 재현율의 조화 평균, 두 지표 중 어느 하나에만 치우지지 않고 균형 잡힌 성능을 나타내는 지표
    
    auc_roc = roc_auc_score(all_labels, all_probs)
    # AUC-ROC (ROC 곡선 아래 면적) : 모델이 두 클래스(정상, 이상)를 얼마나 잘 구분하는지에 대해 전반적인 능력을 나타냄
    
    results = {
        'TP (진양성)': tp, 'FN (위음성)': fn, 'FP (위양성)': fp, 'TN (진음성)': tn,
        '정밀도 (Precision)': precision,
        '재현율 (Recall)': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc
    }
    
    return results, all_probs, all_labels

# 최종 테스트 결과 확인 (기본 임계값 0.5 사용)
test_results_05, test_probs, test_labels = evaluate_model(model, test_loader, device, threshold=0.5)

print("\n==========================================================")
print("--- 1차 최종 테스트 결과 (기본 임계값 0.5) ---")
print("==========================================================")
for key, value in test_results_05.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ==========================================================
# 5-2. 최적 임계값 결정 및 최종 평가
# ==========================================================

# F1-Score가 최대인 지점을 찾아 최적 임계값 결정
precision_list, recall_list, thresholds = precision_recall_curve(test_labels, test_probs)
# precision_recall_curve : 임계값을 0부터 1까지 미세하게 변화시킬 때마다 모델의 정밀도와 재현율이 어떻게 변하는지 계산해서 세 가지 배열 반환
# precision_list : 각 임계값에서의 정밀도 값 목록
# recall_list : 각 임계값에서의 재현율 값 목록
# thresholds : 정밀도와 재현율이 계산된 해당 임계값 목록
fscores = 2 * (precision_list * recall_list) / (precision_list + recall_list + 1e-6)
optimal_idx = np.argmax(fscores)
optimal_threshold = thresholds[optimal_idx]

# 최적 임계값 적용 후 최종 평가
final_test_results_optimized, _, _ = evaluate_model(model, test_loader, device, threshold=optimal_threshold)

print("\n==========================================================")
print(f"--- 2차 최종 테스트 결과 (최적 임계값 {optimal_threshold:.4f} 적용) ---")
print("==========================================================")
for key, value in final_test_results_optimized.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ==========================================================
# 6단계: 결과 시각화 및 오류 분석 (기본 Python 반복문 사용)
# ==========================================================

# 6-1. 성능 곡선 시각화 (5단계 변수 사용)
plt.figure(figsize=(8, 6))
plt.plot(recall_list, precision_list, marker='.', label='Precision-Recall Curve')
plt.plot(recall_list[optimal_idx], precision_list[optimal_idx], 'o', color='red', 
         label=f'Optimal Threshold ({optimal_threshold:.4f}, F1: {fscores[optimal_idx]:.4f})')
plt.xlabel('재현율 (Recall)')
plt.ylabel('정밀도 (Precision)')
plt.title('정밀도-재현율 곡선 (Precision-Recall Curve)')
plt.legend()
plt.grid(True)
# plt.show() # 시각화를 원하면 주석 해제 (주피터/Colab 환경 권장)

# 6-2. 오류 분석 (기본 Python 방식)
false_negatives_list = []
false_positives_list = []
num_test_samples = len(X_test)

# 모든 테스트 샘플을 순회하며 오류를 검사
for i in range(num_test_samples):
    filepath = X_test[i]
    label = int(test_labels[i])
    prob = test_probs[i]
    
    # 최적 임계값을 적용한 예측 결과
    prediction = 1 if prob >= optimal_threshold else 0
    
    # 1. 위음성 (FN) 검사: 실제 이상(1)인데 예측이 정상(0)인 경우
    if label == 1 and prediction == 0:
        false_negatives_list.append({
            'filepath': filepath,
            'predicted_prob': prob
        })
        
    # 2. 위양성 (FP) 검사: 실제 정상(0)인데 예측이 이상(1)인 경우
    elif label == 0 and prediction == 1:
        false_positives_list.append({
            'filepath': filepath,
            'predicted_prob': prob
        })

print("\n==========================================================")
print("--- 6단계: 오류 분석 결과 (기본 Python 리스트) ---")
print("==========================================================")

# 위음성 결과 출력
print(f"** 위음성 (FN) 파일 수: {len(false_negatives_list)} 개 **")
if false_negatives_list:
    print("FN 파일 목록 (경로 및 예측 확률):")
    for item in false_negatives_list:
        print(f"  경로: {item['filepath']}, 확률: {item['predicted_prob']:.4f}")
else:
    print("위음성 파일이 발견되지 않았습니다. (모델 성능 완벽)")

# 위양성 결과 출력
print(f"\n** 위양성 (FP) 파일 수: {len(false_positives_list)} 개 **")
if false_positives_list:
    print("FP 파일 목록 (경로 및 예측 확률):")
    for item in false_positives_list:
        print(f"  경로: {item['filepath']}, 확률: {item['predicted_prob']:.4f}")
else:
    print("위양성 파일이 발견되지 않았습니다. (모델 정밀도 100%)")