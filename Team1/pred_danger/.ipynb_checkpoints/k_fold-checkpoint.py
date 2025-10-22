import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE
import warnings
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from sklearn.metrics import classification_report, roc_auc_score

# ==========================================================
# 클래스 및 함수 정의 (이전과 동일)
# ==========================================================

class CarlaDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        json_files_generator = self.data_dir.glob("**/*.json")
        self.json_files = sorted(
            list(json_files_generator),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float('inf')
        )
        print(f"✅ Found {len(self.json_files)} JSON files across all scenario directories.")

    def load_all_frames(self) -> pd.DataFrame:
        all_records = []
        for json_file in tqdm(self.json_files, desc="Loading frames"):
            try:
                frame_id = int(json_file.stem)
                with open(json_file, 'r') as f:
                    frame_data = json.load(f)

                if 'ego_vehicle' not in frame_data or 'vehicles' not in frame_data:
                    continue

                ego = frame_data['ego_vehicle']
                ego_info = {
                    'frame_id': frame_id, 'ego_id': ego['id'],
                    'ego_x': ego['location']['x'], 'ego_y': ego['location']['y'],
                    'ego_vx': ego['velocity']['x'], 'ego_vy': ego['velocity']['y'],
                    'ego_speed': ego['speed'], 'ego_speed_kmh': ego['speed_kmh'],
                    'ego_yaw': ego['rotation']['yaw']
                }
                
                for vehicle_id, vehicle_info in frame_data['vehicles'].items():
                    vd, di = vehicle_info['vehicle_data'], vehicle_info['dynamic_info']
                    record = {
                        'frame_id': frame_id, 'vehicle_id': int(vehicle_id), 'label': vehicle_info['label'],
                        **ego_info,
                        'vehicle_x': vd['location']['x'], 'vehicle_y': vd['location']['y'],
                        'vehicle_vx': vd['velocity']['x'], 'vehicle_vy': vd['velocity']['y'],
                        'vehicle_speed': vd['speed'], 'vehicle_speed_kmh': vd['speed_kmh'],
                        'vehicle_yaw': vd['rotation']['yaw'],
                        'ego_distance': vehicle_info['ego_distance'], 'min_distance': vehicle_info['min_distance'],
                        'collision_probability': vehicle_info['collision_probability'],
                        'approach_rate': di['approach_rate'], 'is_behind': int(di.get('is_behind', -1)),
                        'critical_distance': di['critical_distance'], 'max_speed': di['max_speed']
                    }
                    all_records.append(record)
            except Exception:
                continue
        
        if not all_records: return pd.DataFrame()

        df = pd.DataFrame(all_records).sort_values(['vehicle_id', 'frame_id']).reset_index(drop=True)
        print(f"\n총 {len(df)} 레코드 로드")
        print(f"    프레임 수: {df['frame_id'].nunique()}, 차량 수: {df['vehicle_id'].nunique()}, 사고 케이스: {df['label'].sum()}")
        return df

class CollisionFeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['relative_vx'] = df['vehicle_vx'] - df['ego_vx']
        df['relative_vy'] = df['vehicle_vy'] - df['ego_vy']
        df['relative_speed'] = np.sqrt(df['relative_vx']**2 + df['relative_vy']**2)
        df['relative_x'] = df['vehicle_x'] - df['ego_x']
        df['relative_y'] = df['vehicle_y'] - df['ego_y']
        
        for vehicle_id in df['vehicle_id'].unique():
            mask = df['vehicle_id'] == vehicle_id
            vehicle_df = df[mask].copy()
            df.loc[mask, 'ego_acceleration'] = vehicle_df['ego_speed'].diff().fillna(0)
            df.loc[mask, 'vehicle_acceleration'] = vehicle_df['vehicle_speed'].diff().fillna(0)
            df.loc[mask, 'distance_change_rate'] = vehicle_df['ego_distance'].diff().fillna(0)
            closing_speed = np.where(vehicle_df['is_behind'] == 0, vehicle_df['ego_speed'] - vehicle_df['vehicle_speed'], vehicle_df['vehicle_speed'] - vehicle_df['ego_speed'])
            ttc = np.where(closing_speed > 0.5, vehicle_df['ego_distance'] / closing_speed, 999.0)
            df.loc[mask, 'ttc'] = np.clip(ttc, 0, 999)
        print("Feature Engineering 완료")
        return df

class SequenceGenerator:
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 60, stride: int = 10, camera_type: str = 'both'):
        self.sequence_length, self.prediction_horizon, self.stride, self.camera_type = sequence_length, prediction_horizon, stride, camera_type
        self.feature_columns = [
            'relative_x', 'relative_y', 'relative_vx', 'relative_vy', 'relative_speed', 'ego_speed', 
            'ego_acceleration', 'vehicle_speed', 'vehicle_acceleration', 'ego_distance', 'min_distance', 
            'approach_rate', 'ttc', 'distance_change_rate', 'collision_probability', 'critical_distance'
        ]
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.camera_type == 'front': df = df[df['is_behind'] == 0].copy()
        elif self.camera_type == 'rear': df = df[df['is_behind'] == 1].copy()
        
        X_list, y_list = [], []
        for vehicle_id in tqdm(df['vehicle_id'].unique(), desc="Creating sequences"):
            vehicle_df = df[df['vehicle_id'] == vehicle_id].sort_values('frame_id').reset_index(drop=True)
            if len(vehicle_df) < self.sequence_length + self.prediction_horizon: continue
            
            features, labels = vehicle_df[self.feature_columns].values, vehicle_df['label'].values
            for i in range(0, len(features) - self.sequence_length - self.prediction_horizon, self.stride):
                future_idx = i + self.sequence_length + self.prediction_horizon - 1
                if future_idx < len(labels):
                    X_list.append(features[i:i + self.sequence_length])
                    y_list.append(labels[future_idx])
        
        X, y = np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)
        print(f"\n시퀀스 생성 완료: Shape: {X.shape}, Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        return X, y

class CollisionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class PatchEmbedding(nn.Module):
    def __init__(self, c_in, patch_len, stride, d_model):
        super().__init__()
        self.patch_proj = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=stride)
    def forward(self, x):
        return self.patch_proj(x.transpose(1, 2)).transpose(1, 2)

class PatchTSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x): return self.encoder(x)

class PatchTSTCollisionPredictor(nn.Module):
    def __init__(self, c_in, seq_len, patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1, num_classes=2):
        super().__init__()
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_embedding = PatchEmbedding(c_in, patch_len, stride, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.n_patches, dropout=dropout)
        self.encoder = PatchTSTEncoder(d_model, n_heads, d_ff, dropout, n_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.n_patches, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.encoder(self.pos_encoding(self.patch_embedding(x))))

class CollisionTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', use_focal_loss=True):
        self.model, self.device = model.to(device), device
        self.criterion = FocalLoss() if use_focal_loss else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total += y_batch.size(0)
            correct += (outputs.max(1)[1] == y_batch).sum().item()
        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch.to(self.device))
                loss = self.criterion(outputs, y_batch.to(self.device))
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs=50, save_dir='./model'):
        best_val_loss, patience_counter, max_patience = float('inf'), 0, 15
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'best_model.pth')
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss, patience_counter = val_loss, 0
                torch.save({'model_state_dict': self.model.state_dict()}, save_path)
            else:
                patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⚠️ Early stopping at epoch {epoch+1}"); break

# ==========================================================
# ✨ 1. evaluate_model 함수 수정: test_loss 계산 추가
# ==========================================================
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    # Loss 계산을 위한 객체 및 변수 추가
    criterion = FocalLoss()
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating", leave=False):
            X_batch_dev, y_batch_dev = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch_dev)
            
            # Loss 계산 추가
            loss = criterion(outputs, y_batch_dev)
            total_test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(outputs.max(1)[1].cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    report = classification_report(all_labels, all_preds, target_names=['No Collision', 'Collision'], digits=4, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    avg_test_loss = total_test_loss / len(test_loader)
    
    # 반환하는 딕셔너리에 test_loss 추가
    return {
        'test_loss': avg_test_loss,
        'accuracy': report['accuracy'], 
        'precision_class1': report['Collision']['precision'],
        'recall_class1': report['Collision']['recall'], 
        'f1_class1': report['Collision']['f1-score'],
        'roc_auc': roc_auc
    }


# ==========================================================
# ✨ 2. 메인 실행 부분 수정: 최종 결과 집계 방식 변경
# ==========================================================
if __name__ == "__main__":
    
    # --- 설정값 ---
    DATA_DIR = "/run/user/1000/gvfs/smb-share:server=10.10.14.211,share=carla_data/_output_extracted"
    N_SPLITS = 5
    EPOCHS = 50
    STRIDE = 2
    
    # --- 폴더 생성 ---
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./k_fold', exist_ok=True)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    # --- 1. 데이터 준비 (전체 데이터셋) ---
    print("=" * 60, "\nSTEP 1: 데이터 로딩")
    df = CarlaDataLoader(DATA_DIR).load_all_frames()
    
    if df.empty:
        print("\n❌ 에러: 로드된 데이터가 없습니다. DATA_DIR 경로를 확인해주세요.")
        exit(1)

    print("\n" + "=" * 60, "\nSTEP 2: Feature Engineering")
    df = CollisionFeatureEngineer.add_features(df)
    
    print("\n" + "=" * 60, "\nSTEP 3: 시퀀스 생성")
    seq_gen = SequenceGenerator(sequence_length=60, prediction_horizon=60, stride=STRIDE, camera_type='front')
    X, y = seq_gen.create_sequences(df)

    if len(X) < N_SPLITS or y.sum() < N_SPLITS:
        print(f"\n❌ 에러: 데이터 샘플(총 {len(X)}개, 사고 {y.sum()}개)이 Fold 수({N_SPLITS})보다 부족하여 교차 검증을 진행할 수 없습니다.")
        exit(1)

    # --- 2. 교차 검증 시작 ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_summaries = []

    model_config = {
        'c_in': X.shape[-1], 'seq_len': X.shape[1], 'patch_len': 16, 'stride': 8,
        'd_model': 128, 'n_heads': 8, 'n_layers': 3, 'd_ff': 256, 'dropout': 0.1, 'num_classes': 2
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}\n🚀 FOLD {fold+1}/{N_SPLITS} 시작\n{'='*60}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(n_samples, n_timesteps * n_features)
        
        n_positive = y_train.sum()
        k_neighbors = min(5, n_positive - 1) if n_positive > 1 else 1

        if k_neighbors >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)
            X_train, y_train = X_train_smote.reshape(-1, n_timesteps, n_features), y_train_smote
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

        train_loader = DataLoader(CollisionDataset(X_train_sub, y_train_sub), batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(CollisionDataset(X_val_sub, y_val_sub), batch_size=64, shuffle=False, num_workers=4)
        test_loader = DataLoader(CollisionDataset(X_test, y_test), batch_size=64, shuffle=False, num_workers=4)

        model = PatchTSTCollisionPredictor(**model_config)
        trainer = CollisionTrainer(model)
        trainer.train(train_loader, val_loader, epochs=EPOCHS, save_dir=f'./model/fold_{fold+1}')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_model_path = f'./model/fold_{fold+1}/best_model.pth'
        model.load_state_dict(torch.load(best_model_path, weights_only=False)['model_state_dict'])
        model = model.to(device)
        
        fold_summary = evaluate_model(model, test_loader, device)
        fold_summaries.append(fold_summary)
        print(f"Fold {fold+1} 결과: {pd.Series(fold_summary).to_string()}")

    # --- 3. 최종 결과 집계 및 저장 ---
    print(f"\n{'='*60}\n📊 5-Fold 교차 검증 최종 결과\n{'='*60}")

    # 1. 각 Fold의 결과를 DataFrame으로 만듭니다.
    cv_results_df = pd.DataFrame(fold_summaries)
    cv_results_df.index = [f'fold_{i+1}' for i in range(N_SPLITS)]
    
    # 2. 평균(mean)과 표준편차(std) 행을 추가합니다.
    cv_results_df.loc['mean'] = cv_results_df.mean()
    cv_results_df.loc['std'] = cv_results_df.std()
    
    # 3. 소수점 3자리로 포맷팅합니다.
    final_summary_df = cv_results_df.round(3)
    
    print("성능 요약:")
    print(final_summary_df)

    save_path = './k_fold/cv_performance_summary.csv'
    final_summary_df.to_csv(save_path)
    print(f"\n✅ 최종 교차 검증 결과가 '{save_path}'에 저장되었습니다.")