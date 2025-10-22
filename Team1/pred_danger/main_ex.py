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
from typing import Tuple
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import random
# ==========================================================
# 클래스 및 기본 함수 정의 (분류 모델 버전)
# ==========================================================

class CarlaDataLoader: # (수정 없음)
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        json_files_generator = self.data_dir.glob("**/*.json")
        self.json_files = sorted(list(json_files_generator), key=lambda p: int(p.stem) if p.stem.isdigit() else float('inf'))
        print(f"✅ Found {len(self.json_files)} JSON files.")
    def load_all_frames(self) -> pd.DataFrame:
        all_records = []
        for json_file in tqdm(self.json_files, desc="Loading frames"):
            try:
                frame_id = int(json_file.stem);
                with open(json_file, 'r') as f: data = json.load(f)
                if 'ego_vehicle' not in data or 'vehicles' not in data: continue
                ego = data['ego_vehicle']; ego_info = {'frame_id': frame_id, 'ego_id': ego['id'],'ego_x': ego['location']['x'], 'ego_y': ego['location']['y'],'ego_vx': ego['velocity']['x'], 'ego_vy': ego['velocity']['y'],'ego_speed': ego['speed'], 'ego_speed_kmh': ego['speed_kmh'],'ego_yaw': ego['rotation']['yaw']}
                for vid, v_info in data['vehicles'].items():
                    vd, di = v_info['vehicle_data'], v_info['dynamic_info']; record = {'frame_id': frame_id, 'vehicle_id': int(vid), 'label': v_info['label'],**ego_info,'vehicle_x': vd['location']['x'], 'vehicle_y': vd['location']['y'],'vehicle_vx': vd['velocity']['x'], 'vehicle_vy': vd['velocity']['y'],'vehicle_speed': vd['speed'], 'vehicle_speed_kmh': vd['speed_kmh'],'vehicle_yaw': vd['rotation']['yaw'],'ego_distance': v_info['ego_distance'], 'min_distance': v_info['min_distance'],'collision_probability': v_info['collision_probability'],'approach_rate': di['approach_rate'], 'is_behind': int(di.get('is_behind', -1)),'critical_distance': di['critical_distance'], 'max_speed': di['max_speed']}; all_records.append(record)
            except Exception: continue
        if not all_records: return pd.DataFrame()
        df = pd.DataFrame(all_records).sort_values(['vehicle_id', 'frame_id']).reset_index(drop=True)
        print(f"총 {len(df)} 레코드 로드 (프레임: {df['frame_id'].nunique()}, 차량: {df['vehicle_id'].nunique()}, 사고: {df['label'].sum()})")
        return df

class CollisionFeatureEngineer: # (수정 없음)
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(); df['relative_vx'] = df['vehicle_vx'] - df['ego_vx']; df['relative_vy'] = df['vehicle_vy'] - df['ego_vy']; df['relative_speed'] = np.sqrt(df['relative_vx']**2 + df['relative_vy']**2); df['relative_x'] = df['vehicle_x'] - df['ego_x']; df['relative_y'] = df['vehicle_y'] - df['ego_y']
        for vehicle_id in tqdm(df['vehicle_id'].unique(), desc="Feature Engineering"):
            mask = df['vehicle_id'] == vehicle_id; v_df = df[mask].copy(); df.loc[mask, 'ego_acceleration'] = v_df['ego_speed'].diff().fillna(0); df.loc[mask, 'vehicle_acceleration'] = v_df['vehicle_speed'].diff().fillna(0); df.loc[mask, 'distance_change_rate'] = v_df['ego_distance'].diff().fillna(0); closing_speed = np.where(v_df['is_behind']==0, v_df['ego_speed']-v_df['vehicle_speed'], v_df['vehicle_speed']-v_df['ego_speed']); ttc = np.where(closing_speed > 0.5, v_df['ego_distance'] / closing_speed, 999.0); df.loc[mask, 'ttc'] = np.clip(ttc, 0, 999)
        print("Feature Engineering 완료")
        return df

# SequenceGenerator 클래스 전체를 이 코드로 교체하세요.

class SequenceGenerator:
    def __init__(self, seq_len=60, pred_horizon=60, stride=10, cam_type='both', pad_mode='repeat_first'):
        self.seq_len, self.pred_horizon, self.stride, self.cam_type = seq_len, pred_horizon, stride, cam_type
        self.pad_mode = pad_mode  # 'repeat_first' or 'zero'
        self.features = ['relative_x', 'relative_y', 'relative_vx', 'relative_vy', 'relative_speed', 'ego_speed', 'ego_acceleration', 'vehicle_speed', 'vehicle_acceleration', 'ego_distance', 'min_distance', 'approach_rate', 'ttc', 'distance_change_rate', 'collision_probability', 'critical_distance']
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.cam_type == 'front': df = df[df['is_behind'] == 0].copy()
        elif self.cam_type == 'rear': df = df[df['is_behind'] == 1].copy()
        
        positive_sequences = [] # 사고(1)로 이어지는 시퀀스
        negative_sequences = [] # 사고가 아닌(0) 시퀀스

        for vid in tqdm(df['vehicle_id'].unique(), desc=f"Analyzing trajectories for '{self.cam_type}'"):
            v_df = df[df['vehicle_id'] == vid].sort_values('frame_id').reset_index(drop=True)
            
            # 이 차량의 모든 충돌 지점(프레임 인덱스)을 찾음
            collision_indices = v_df.index[v_df['label'] == 1].tolist()
            
            # --- 1. Positive 샘플 생성 (사고 직전 데이터) ---
            for col_idx in collision_indices:
                # 사고 발생 'pred_horizon' 프레임 전부터 'seq_len' 길이의 데이터를 추출
                start_idx = col_idx - self.pred_horizon - self.seq_len + 1
                end_idx = col_idx - self.pred_horizon + 1

                # 앞부분 데이터가 부족한 경우도 패딩으로 보완하여 포함
                if end_idx <= 0:
                    # 사고 예측 시점 이전에 사용 가능한 프레임이 전혀 없는 경우: 패딩으로 구성
                    if len(v_df) > 0:
                        if self.pad_mode == 'zero':
                            pad = np.zeros((self.seq_len, len(self.features)), dtype=np.float32)
                        else:
                            first = v_df.loc[0, self.features].values
                            pad = np.tile(first, (self.seq_len, 1))
                        positive_sequences.append({'X': pad, 'y': 1})
                    continue

                if start_idx >= 0:
                    sequence_x = v_df.loc[start_idx:end_idx-1, self.features].values
                    if sequence_x.shape[0] == self.seq_len:
                        positive_sequences.append({'X': sequence_x, 'y': 1})
                else:
                    # 사용 가능한 구간만 가져오고, 부족분은 첫 값으로 패딩
                    available = v_df.loc[0:end_idx-1, self.features].values
                    if available.shape[0] > 0:
                        pad_len = self.seq_len - available.shape[0]
                        if pad_len > 0:
                            if self.pad_mode == 'zero':
                                pad = np.zeros((pad_len, available.shape[1]), dtype=available.dtype)
                            else:
                                pad = np.repeat(available[0:1, :], pad_len, axis=0)
                            sequence_x = np.vstack([pad, available])
                        else:
                            sequence_x = available[-self.seq_len:]
                        positive_sequences.append({'X': sequence_x, 'y': 1})

            # --- 2. Negative 샘플 후보군 생성 ---
            # (stride를 사용하여 전체 데이터에서 샘플링)
            features = v_df[self.features].values
            labels = v_df['label'].values
            # 뒤쪽 가장자리도 자르지 않도록, 예측 지평선을 가능한 범위만 확인하여 포함
            last_i = len(features) - self.seq_len
            for i in range(0, last_i + 1, self.stride):
                end_idx = i + self.seq_len
                future_end = min(end_idx + self.pred_horizon, len(labels))
                # 사용 가능한 미래 구간(빈 구간 포함)에서 충돌이 없으면 Negative로 포함
                if not np.any(labels[end_idx:future_end]):
                    negative_sequences.append({'X': features[i:end_idx], 'y': 0})

        if not positive_sequences:
            print("❌ 경고: 사고(label=1) 데이터를 찾을 수 없어 학습 데이터셋을 만들 수 없습니다.")
            return np.array([]), np.array([])
        # --- 3. 데이터 균형 조정 제거: 모든 Negative 샘플 사용 ---
        print(f"\n데이터 샘플링 완료 (다운샘플링 없이 전체 사용):")
        print(f"  - Positive (사고 직전) 샘플: {len(positive_sequences)}개")
        print(f"  - Negative (안전) 샘플: {len(negative_sequences)}개")

        final_sequences = positive_sequences + negative_sequences
        random.shuffle(final_sequences)
        
        X = np.array([seq['X'] for seq in final_sequences], dtype=np.float32)
        y = np.array([seq['y'] for seq in final_sequences], dtype=np.int64)
        
        print(f"최종 시퀀스 생성 완료 (Shape: {X.shape}, Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()})")
        return X, y

class CollisionDataset(Dataset): # ✨ 수정됨: y를 LongTensor로 되돌림
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) # FloatTensor -> LongTensor
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class FocalLoss(nn.Module): # (수정 없음)
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, i, t):
        ce = F.cross_entropy(i, t, reduction='none'); pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt)**self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class PatchTSTCollisionPredictor(nn.Module): # ✨ 수정됨: 분류 모델로 되돌림
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
        # ✨ 출력층을 2개로 바꾸고, Sigmoid 제거
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(d_model * n_patches, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, x):
        x = self.patch_embedding(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x + self.pos_encoding)
        x = self.encoder(x)
        return self.classifier(x)

class CollisionTrainer: # ✨ 수정됨: 분류용 + Early Stopping 설정 지원
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=1e-4, weight_decay=0.01,
                 es_patience=15, es_min_delta=0.0, es_monitor='val_loss', es_mode='min'):
        self.model, self.device = model.to(device), device
        self.criterion = FocalLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        # Early stopping 설정
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.es_monitor = es_monitor  # 'val_loss' 또는 'f1_class1'
        self.es_mode = es_mode        # 'min' 또는 'max'
    def train(self, train_loader, val_loader, epochs, save_path):
        best_score = None
        es_counter = 0
        validation_criterion = nn.CrossEntropyLoss() # 검증은 일반 CrossEntropy 사용
        for epoch in range(epochs):
            self.model.train(); total_loss, correct, total = 0,0,0
            for X_b, y_b in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
                X_b, y_b = X_b.to(self.device), y_b.to(self.device); self.optimizer.zero_grad()
                out = self.model(X_b); loss = self.criterion(out, y_b); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
                total_loss+=loss.item(); total+=y_b.size(0); correct+=(out.max(1)[1]==y_b).sum().item()
            train_loss, train_acc = total_loss/len(train_loader), 100.*correct/total
            self.model.eval(); val_loss = 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = self.model(X_b.to(self.device)); val_loss+=validation_criterion(out, y_b.to(self.device)).item()
            val_loss /= len(val_loader); self.scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}")
            # 모니터링 지표 계산
            if self.es_monitor == 'val_loss':
                current = val_loss
            else:
                # f1_class1 모니터링: 간단한 샘플 평가 (validation loader 기준)
                y_true, y_pred = [], []
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        logits = self.model(X_b.to(self.device))
                        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                        preds = (probs > 0.5).astype(int)  # 모니터링 고정 임계값 0.5
                        y_true.extend(y_b.numpy()); y_pred.extend(preds)
                rep = classification_report(y_true, y_pred, target_names=['No Collision','Collision'], output_dict=True, zero_division=0)
                current = rep['Collision']['f1-score']

            improved = False
            if best_score is None:
                improved = True
            else:
                delta = (current - best_score) if self.es_mode == 'max' else (best_score - current)
                improved = delta > self.es_min_delta

            if improved:
                best_score = current
                es_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                es_counter += 1
            if es_counter >= self.es_patience:
                print("Early stopping.")
                break

# evaluate_model 함수 전체를 이 코드로 교체하세요.

def evaluate_model(model, loader, device, threshold=0.5): # ✨ 1. threshold 파라미터 추가
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    criterion = FocalLoss()
    loss = 0
    with torch.no_grad():
        for X_b, y_b in tqdm(loader, desc=f"Evaluating with threshold {threshold}", leave=False):
            out = model(X_b.to(device))
            loss += criterion(out, y_b.to(device)).item()
            probs = torch.softmax(out, dim=1)
            # ✨ 2. 예측값(preds) 대신 확률(probs)을 먼저 수집
            all_labels.extend(y_b.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # ✨ 3. 루프가 끝난 후, 새로운 임계값을 적용하여 최종 예측 생성
    all_probs_np = np.array(all_probs)
    all_preds = (all_probs_np > threshold).astype(int)

    # (이하 평가 지표 계산은 동일)
    report = classification_report(all_labels, all_preds, target_names=['No Collision', 'Collision'], digits=4, output_dict=True, zero_division=0)
    roc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    return {'test_loss': loss/len(loader), 'accuracy': report['accuracy'], 'precision_class1': report['Collision']['precision'], 'recall_class1': report['Collision']['recall'], 'f1_class1': report['Collision']['f1-score'], 'roc_auc': roc}

def sweep_threshold(model, loader, device, thresholds=None):
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]
    results = []
    # 한번만 forward하여 확률 캐시
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            out = model(X_b.to(device))
            probs = torch.softmax(out, dim=1)
            all_labels.extend(y_b.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    for th in thresholds:
        preds = (all_probs > th).astype(int)
        rep = classification_report(all_labels, preds, target_names=['No Collision', 'Collision'], digits=4, output_dict=True, zero_division=0)
        results.append({'threshold': th, 'accuracy': rep['accuracy'], 'precision_class1': rep['Collision']['precision'], 'recall_class1': rep['Collision']['recall'], 'f1_class1': rep['Collision']['f1-score']})
    # f1_class1 최대인 threshold 선택
    best = max(results, key=lambda r: r['f1_class1']) if results else {'threshold': 0.5}
    return best, results
# ==========================================================
# 메인 실행 함수 (분류에 맞게 수정됨)
# ==========================================================
def run_process_for_camera(df: pd.DataFrame, camera_type: str, config: dict):
    
    print(f"\n{'='*80}\n🚀 STARTING PROCESS FOR: {camera_type.upper()} CAMERA\n{'='*80}")
    
    seq_gen = SequenceGenerator(seq_len=60, pred_horizon=60, stride=config['STRIDE'], cam_type=camera_type)
    X, y = seq_gen.create_sequences(df)
    if len(X) == 0: print(f"❌ ERROR: No data for {camera_type.upper()} camera. Skipping."); return
    
    model_config = {'c_in':X.shape[-1], 'seq_len':X.shape[1], 'patch_len':16, 'stride':8, 'd_model':128, 'n_heads':8, 'n_layers':3, 'd_ff':256, 'dropout':0.1, 'n_classes':2}

    # --- 1. K-Fold CV 성능 평가 ---
    if len(X) < config['N_SPLITS'] or (len(y) > 0 and y.sum() < config['N_SPLITS']):
        print(f"❌ ERROR: Not enough data for {config['N_SPLITS']}-Fold CV. Skipping CV.")
    else:
        print(f"\n{'-'*60}\nSTEP 1: Performing K-Fold CV for {camera_type.upper()} Camera\n{'-'*60}")
        skf = StratifiedKFold(n_splits=config['N_SPLITS'], shuffle=True, random_state=42) # ✨ StratifiedKFold 사용
        fold_summaries = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n➡️ FOLD {fold+1}/{config['N_SPLITS']}")
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            
            # ✨ SMOTE 다시 적용
            n_samples, n_timesteps, n_features = X_train.shape
            if y_train.sum() > 1:
                k = min(5, y_train.sum() - 1)
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_train_reshaped = X_train.reshape(n_samples, n_timesteps * n_features)
                X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)
                X_train, y_train = X_train_smote.reshape(-1, n_timesteps, n_features), y_train_smote

            scaler_cv = StandardScaler()
            X_train = scaler_cv.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
            X_test = scaler_cv.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
            
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42) # ✨ stratify 다시 사용
            
            cv_model = PatchTSTCollisionPredictor(**model_config) # ✨ Predictor 모델 사용
            cv_trainer = CollisionTrainer(
                cv_model,
                es_patience=config.get('ES_PATIENCE', 15),
                es_min_delta=config.get('ES_MIN_DELTA', 0.0),
                es_monitor=config.get('ES_MONITOR', 'val_loss'),
                es_mode='min' if config.get('ES_MONITOR', 'val_loss')=='val_loss' else 'max'
            )
            temp_path = f"./model/temp_model_{camera_type}_fold_{fold+1}.pth"
            cv_trainer.train(DataLoader(CollisionDataset(X_train_sub, y_train_sub), 64, shuffle=True), DataLoader(CollisionDataset(X_val_sub, y_val_sub), 64), config['EPOCHS'], temp_path)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cv_model.load_state_dict(torch.load(temp_path, weights_only=False))
            cv_model = cv_model.to(device)
            
            test_loader = DataLoader(CollisionDataset(X_test, y_test), 64)
            # threshold sweep으로 최적 임계값 탐색
            best_th, _ = sweep_threshold(cv_model, test_loader, device)
            fold_summary = evaluate_model(cv_model, test_loader, device, threshold=best_th['threshold'])
            fold_summary['best_threshold'] = best_th['threshold']
            fold_summaries.append(fold_summary); print(f"Fold {fold+1} Result: {pd.Series(fold_summary).to_string()}")
            os.remove(temp_path)

        print(f"\n📊 Final {config['N_SPLITS']}-Fold CV Results for {camera_type.upper()} Camera")
        cv_results_df = pd.DataFrame(fold_summaries); cv_results_df.index = [f'fold_{i+1}' for i in range(config['N_SPLITS'])]
        cv_results_df.loc['mean'] = cv_results_df.mean(); cv_results_df.loc['std'] = cv_results_df.std()
        final_summary_df = cv_results_df.round(4)
        print("Performance Summary:\n", final_summary_df)
        save_path = f'./k_fold/cv_performance_summary_{camera_type}_classification.csv' # 파일 이름 변경
        final_summary_df.to_csv(save_path); print(f"✅ CV results saved to '{save_path}'")
        # CV에서의 최적 threshold 평균 저장
        if 'best_threshold' in final_summary_df.columns:
            mean_best_th = final_summary_df.loc['mean', 'best_threshold'] if 'mean' in final_summary_df.index else float(np.nan)
            with open(f'./model/best_threshold_{camera_type}.json', 'w') as f:
                json.dump({'camera_type': camera_type, 'best_threshold_cv_mean': float(mean_best_th)}, f)
            print(f"✅ Saved best threshold (CV mean) for {camera_type.upper()} -> {mean_best_th}")
        
    # --- 2. 최종 모델 학습 및 저장 ---
    print(f"\n{'-'*60}\nSTEP 2: Training Final Model for {camera_type.upper()} Camera\n{'-'*60}")
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # ✨ stratify 다시 사용
    
    # ✨ SMOTE 다시 적용
    n_samples, n_timesteps, n_features = X_train_full.shape
    if y_train_full.sum() > 1:
        k = min(5, y_train_full.sum() - 1)
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_reshaped = X_train_full.reshape(n_samples, n_timesteps * n_features)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train_full)
        X_train_full, y_train_full = X_train_smote.reshape(-1, n_timesteps, n_features), y_train_smote
    
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full.reshape(-1, n_features)).reshape(X_train_full.shape)
    X_val_full = scaler.transform(X_val_full.reshape(-1, n_features)).reshape(X_val_full.shape)

    final_model_path = f"./model/best_model_{camera_type}_classification.pth" # 파일 이름 변경
    final_scaler_path = f"./model/scaler_{camera_type}_classification.pkl" # 파일 이름 변경
    with open(final_scaler_path, 'wb') as f: pickle.dump(scaler, f)
    
    final_model = PatchTSTCollisionPredictor(**model_config) # ✨ Predictor 모델 사용
    trainer = CollisionTrainer(
        final_model,
        es_patience=config.get('ES_PATIENCE', 15),
        es_min_delta=config.get('ES_MIN_DELTA', 0.0),
        es_monitor=config.get('ES_MONITOR', 'val_loss'),
        es_mode='min' if config.get('ES_MONITOR', 'val_loss')=='val_loss' else 'max'
    )
    trainer.train(DataLoader(CollisionDataset(X_train_full, y_train_full), 64, shuffle=True), DataLoader(CollisionDataset(X_val_full, y_val_full), 64), config['EPOCHS'], final_model_path)
    print(f"✅ Final model for {camera_type.upper()} saved to '{final_model_path}'")
    print(f"✅ Scaler for {camera_type.upper()} saved to '{final_scaler_path}'")

    # 최종 모델에 대해 validation 기준으로 최적 threshold 저장
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_model.load_state_dict(torch.load(final_model_path, weights_only=False))
    final_model = final_model.to(device)
    best_th, _ = sweep_threshold(final_model, DataLoader(CollisionDataset(X_val_full, y_val_full), 64), device)
    with open(f'./model/best_threshold_{camera_type}_final.json', 'w') as f:
        json.dump({'camera_type': camera_type, 'best_threshold_final': float(best_th['threshold'])}, f)
    print(f"✅ Saved best threshold (final val) for {camera_type.upper()} -> {best_th['threshold']}")


# ==========================================================
# 메인 실행 블록
# ==========================================================
if __name__ == "__main__":
    CONFIG = {
        "DATA_DIR": "/run/user/1000/gvfs/smb-share:server=10.10.14.211,share=carla_datasets/_output_extracted",
        "N_SPLITS": 5, "EPOCHS": 100, "STRIDE": 2,
        # Early Stopping 설정
        "ES_PATIENCE": 15,
        "ES_MIN_DELTA": 0.0,
        # 'val_loss' 또는 'f1_class1' 중 선택
        "ES_MONITOR": "val_loss"
    }
    
    os.makedirs('./k_fold', exist_ok=True); os.makedirs('./model', exist_ok=True)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    print("=" * 80, "\nSTEP 1: Loading and Preprocessing Data\n" + "=" * 80)
    df = CarlaDataLoader(CONFIG['DATA_DIR']).load_all_frames()
    if df.empty: print("❌ ERROR: No data loaded."); exit(1)
    df = CollisionFeatureEngineer.add_features(df)
    
    run_process_for_camera(df=df, camera_type='front', config=CONFIG)
    run_process_for_camera(df=df, camera_type='rear', config=CONFIG)

    print(f"\n{'='*80}\n🎉 ALL PROCESSES COMPLETED!\n{'='*80}")