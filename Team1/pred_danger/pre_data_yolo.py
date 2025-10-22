import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import pandas as pd # ⚠️ 'pd' is not defined 오류 해결

# ==========================================================
# 설정값
# ==========================================================
# 1. 원본 CARLA 데이터가 있는 최상위 폴더 경로
BASE_DATA_DIR = Path("/run/user/1000/gvfs/smb-share:server=10.10.14.211,share=carla_data/_output_extracted")
# 2. 최종 YOLO 데이터셋이 생성될 폴더 이름
OUTPUT_DIR = Path("./yolo_carla_dataset")
# 3. 클래스 정의 (모든 차량을 'vehicle' 하나로 통일)
CLASS_MAPPING = {"vehicle": 0}
CLASS_NAMES = list(CLASS_MAPPING.keys())

# 4. PatchTST와 동일한 윈도우 크기 (사고 발생 전 몇 프레임을 '중요'하게 볼 것인가)
SEQ_LEN = 60
PRED_HORIZON = 60
WINDOW_SIZE = SEQ_LEN + PRED_HORIZON

# 5. 데이터 분할 및 샘플링 비율
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# Negative 샘플을 Positive 샘플 수의 N배만큼 추출 (예: 2.0은 2배)
NEGATIVE_SAMPLING_RATIO = 2.0 

# ==========================================================
# 유틸리티 함수
# ==========================================================
def convert_bbox_to_yolo(bbox):
    """CARLA bbox [xmin, ymin, xmax, ymax]를 YOLO 형식으로 변환"""
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)
    return x_center, y_center, width, height

# ==========================================================
# 메인 데이터 처리 함수
# ==========================================================
def process_data():
    if OUTPUT_DIR.exists():
        print(f"경고: 기존 '{OUTPUT_DIR}' 폴더를 삭제하고 다시 생성합니다.")
        shutil.rmtree(OUTPUT_DIR)
        
    vehicle_class_id = CLASS_MAPPING["vehicle"]

    print("➡️ STEP 1: 모든 시나리오를 분석하여 '사고 직전' 중요 구간을 찾습니다...")
    important_frames = set()
    
    scenario_dirs = [d for d in BASE_DATA_DIR.iterdir() if d.is_dir()]

    for scenario_dir in tqdm(scenario_dirs, desc="Finding collision windows"):
        label_dir = scenario_dir / "ground_truth_labels"
        if not label_dir.exists(): continue
        
        try:
            json_files = list(label_dir.glob("*.json"))
            if not json_files: continue
            df_for_scan = pd.DataFrame([json.load(open(f)) for f in json_files])
        except (json.JSONDecodeError, ValueError) as e:
            print(f"경고: {scenario_dir.name} 에서 JSON 파일을 읽는 중 오류({e})가 발생하여 건너뜁니다.")
            continue

        if 'vehicles' not in df_for_scan.columns: continue

        all_vehicles_data = []
        for index, row in df_for_scan.iterrows():
            frame_id = row.get('frame_id')
            if frame_id is None: continue
            
            vehicles_dict = row.get('vehicles')
            if isinstance(vehicles_dict, dict):
                for vehicle_id, v_info in vehicles_dict.items():
                    all_vehicles_data.append({
                        'frame_id': frame_id, 
                        'vehicle_id': int(vehicle_id), 
                        'label': v_info.get('label', 0)
                    })
        
        if not all_vehicles_data: continue
        
        df_labels = pd.DataFrame(all_vehicles_data)

        for vehicle_id in df_labels['vehicle_id'].unique():
            v_df = df_labels[df_labels['vehicle_id'] == vehicle_id].sort_values('frame_id')
            collision_frames = v_df[v_df['label'] == 1]['frame_id'].tolist()
            
            for frame in collision_frames:
                # ✨✨✨ 여기가 수정된 부분! ✨✨✨
                # frame 변수를 int()로 감싸서 정수로 변환합니다.
                int_frame = int(frame)
                for i in range(int_frame - WINDOW_SIZE + 1, int_frame + 1):
                # ✨✨✨ 수정 끝 ✨✨✨
                    if i >= 0:
                        important_frames.add((str(scenario_dir), str(i).zfill(6)))

    print(f"✅ 총 {len(important_frames)}개의 프레임을 중요 구간으로 식별했습니다.")

    # ... (이하 코드는 이전과 동일합니다) ...

    print("\n➡️ STEP 2: 이미지-라벨 쌍을 수집하고 중요/일반 데이터로 분류합니다...")
    positive_pairs = []
    negative_pairs = []

    for scenario_dir in tqdm(scenario_dirs, desc="Collecting data pairs"):
        label_dir = scenario_dir / "ground_truth_labels"
        front_img_dir = scenario_dir / "Front"
        rear_img_dir = scenario_dir / "Rear"
        
        if not label_dir.exists(): continue

        for json_path in label_dir.glob("*.json"):
            frame_id_str = json_path.stem
            
            with open(json_path, 'r') as f: data = json.load(f)

            def create_pair(img_path, bbox_key):
                if img_path.exists():
                    labels = []
                    for v_info in data.get('vehicles', {}).values():
                        v_data = v_info.get('vehicle_data', {})
                        if v_data.get(bbox_key):
                            yolo_bbox = convert_bbox_to_yolo(v_data[bbox_key])
                            labels.append(f"{vehicle_class_id} {' '.join(map(str, yolo_bbox))}")
                    if labels:
                        return {"image": img_path, "labels": labels}
                return None

            pair_front = create_pair(front_img_dir / f"{frame_id_str}.png", 'bbox2d_front')
            pair_rear = create_pair(rear_img_dir / f"{frame_id_str}.png", 'bbox2d_rear')

            is_important = (str(scenario_dir), frame_id_str) in important_frames
            
            if pair_front: (positive_pairs if is_important else negative_pairs).append(pair_front)
            if pair_rear: (positive_pairs if is_important else negative_pairs).append(pair_rear)

    print("\n➡️ STEP 3: 데이터셋 균형을 맞추기 위해 샘플링을 진행합니다...")
    
    if not positive_pairs:
        print("❌ 에러: 중요 구간(Positive) 데이터를 찾지 못했습니다. 데이터셋을 생성할 수 없습니다.")
        return

    num_neg_samples = min(len(negative_pairs), int(len(positive_pairs) * NEGATIVE_SAMPLING_RATIO))
    sampled_negative_pairs = random.sample(negative_pairs, k=num_neg_samples)

    print(f"  - Positive (사고 직전) 샘플: {len(positive_pairs)}개")
    print(f"  - Negative (안전) 샘플: {len(sampled_negative_pairs)}개 (총 {len(negative_pairs)}개 중)")

    all_data_pairs = positive_pairs + sampled_negative_pairs
    random.shuffle(all_data_pairs)
    
    print(f"✅ 총 {len(all_data_pairs)}개의 이미지로 최종 데이터셋을 구성합니다.")
    
    print("\n➡️ STEP 4: 데이터를 train/valid/test 세트로 분할하고 파일로 저장합니다...")
    train_count = int(len(all_data_pairs) * TRAIN_RATIO)
    valid_count = int(len(all_data_pairs) * VALID_RATIO)
    
    splits = {
        "train": all_data_pairs[:train_count],
        "valid": all_data_pairs[train_count : train_count + valid_count],
        "test": all_data_pairs[train_count + valid_count :]
    }

    for split_name, data_list in splits.items():
        img_dir = OUTPUT_DIR / split_name / "images"
        lbl_dir = OUTPUT_DIR / split_name / "labels"
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
        
        for item in tqdm(data_list, desc=f"Writing {split_name} set"):
            img_path = item["image"]
            labels = item["labels"]
            shutil.copy(img_path, img_dir / img_path.name)
            label_path = lbl_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(labels))

    print("\n➡️ STEP 5: 'data.yaml' 설정 파일을 생성합니다...")
    yaml_content = f"""
train: {os.path.abspath(OUTPUT_DIR / 'train' / 'images')}
val: {os.path.abspath(OUTPUT_DIR / 'valid' / 'images')}
test: {os.path.abspath(OUTPUT_DIR / 'test' / 'images')}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / "data.yaml", 'w') as f: f.write(yaml_content)
    print(f"\n🎉 모든 준비가 완료되었습니다! '{OUTPUT_DIR}' 폴더를 확인하세요.")

# ==========================================================
# 메인 실행 블록
# ==========================================================
if __name__ == "__main__":
    process_data()