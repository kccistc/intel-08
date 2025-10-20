import sys
sys.path.append('./densefuse_pytorch')

import cv2
import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
import json

# 사용자 정의 모델 및 유틸리티 임포트
from densefuse_pytorch.net import DenseFuse_net

# --- 캘리브레이션 파라미터 로드 함수 ---
def load_calibration_params(file_path):
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        map1_rgb = np.array(params['map1_rgb'], dtype=np.float32)
        map2_rgb = np.array(params['map2_rgb'], dtype=np.float32)
        map1_ir = np.array(params['map1_ir'], dtype=np.float32)
        map2_ir = np.array(params['map2_ir'], dtype=np.float32)
        return map1_rgb, map2_rgb, map1_ir, map2_ir
    except FileNotFoundError:
        print(f"Error: Calibration file not found at {file_path}")
        return None, None, None, None

# --- dcp방식 dehazing + 저조도 개선 함수 ---
def enhance_low_light(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def dehaze(image):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dark_channel = cv2.erode(min_channel, kernel)
    A = np.max(dark_channel)
    t = 1 - 0.95 * dark_channel / A
    t = np.clip(t, 0.2, 1)
    J = np.empty_like(image, dtype=np.float32)
    for c in range(3):
        J[:,:,c] = (image[:,:,c].astype(np.float32) - A) / t + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    
    return J

# --- 모델 로드 및 전처리 설정 ---
dense_model_path = './densefuse_pytorch/models/densefuse_gray.model' 
dense_model = DenseFuse_net(input_nc=3, output_nc=1).to('cuda')

checkpoint_dense = torch.load(dense_model_path, map_location='cuda')
# 체크포인트가 dict인지, state_dict인지 확인 후 로드
if isinstance(checkpoint_dense, dict) and 'model_state_dict' in checkpoint_dense:
    state_dict = checkpoint_dense['model_state_dict']
else:
    state_dict = checkpoint_dense

conv1_weight_1ch = state_dict['conv1.conv2d.weight']
conv1_weight_3ch = conv1_weight_1ch.repeat(1, 3, 1, 1)
state_dict['conv1.conv2d.weight'] = conv1_weight_3ch

dense_model.load_state_dict(state_dict)
dense_model.eval()

to_tensor = ToTensor()

MODEL_INPUT_SIZE = (512, 512)

# --- 캘리브레이션 파라미터 로드 ---
calib_file = 'calib_params.json' 
map1_rgb, map2_rgb, map1_ir, map2_ir = load_calibration_params(calib_file)

if map1_ir is None:
    exit()

# --- 카메라 설정 ---
cap_rgb = cv2.VideoCapture(0)  # RGB 카메라
cap_ir = cv2.VideoCapture(2)   # IR 카메라 (장치 번호 확인 필요)

if not cap_rgb.isOpened() or not cap_ir.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# --- 실시간 처리 루프 ---
while True:
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_ir, frame_ir = cap_ir.read()

    if not ret_rgb or not ret_ir:
        break

    original_h, original_w, _ = frame_rgb.shape

    # 1. IR 영상 정합 및 리사이즈
    aligned_ir = cv2.remap(frame_ir, map1_ir, map2_ir, cv2.INTER_LINEAR)
    aligned_ir_resized = cv2.resize(aligned_ir, MODEL_INPUT_SIZE)

    # 2. RGB 영상 저조도 개선 및 Dehazing
    rgb_input = cv2.resize(frame_rgb, MODEL_INPUT_SIZE)
    enhanced = enhance_low_light(rgb_input)
    dehazed_rgb = dehaze(enhanced)
    # 노이즈 제거
    dehazed_rgb = cv2.fastNlMeansDenoisingColored(dehazed_rgb, None, 6, 10, 7, 21)

    # 한 번만 RGB -> 텐서 변환
    rgb_input_tensor = to_tensor(cv2.cvtColor(dehazed_rgb, cv2.COLOR_BGR2RGB)).unsqueeze(0).to('cuda')

    # 3. IR 이미지는 이미 그레이스케일로 처리됨
    ir_gray = cv2.cvtColor(aligned_ir_resized, cv2.COLOR_BGR2GRAY)
    ir_input = to_tensor(ir_gray).to('cuda').repeat(3, 1, 1)  # 3채널 확장
    
    # 4. DenseFuse 실행
    with torch.no_grad():
        features_rgb = dense_model.encoder(rgb_input_tensor)[0]
        features_ir = dense_model.encoder(ir_input.unsqueeze(0))[0]  # 결과: [1, 64, H, W]

        fused_features_list = dense_model.fusion([features_rgb], [features_ir], strategy_type='addition')
        input_decoder = fused_features_list[0]  # 결과: [1, 64, H, W]

        fused_tensor = dense_model(input_decoder)

    # 5. 결과 후처리
    fused_np = fused_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    fused_uint8 = np.clip(fused_np * 255, 0, 255).astype(np.uint8)

    # 밝기/대비 조절
    fused_adjusted = cv2.convertScaleAbs(fused_uint8, alpha=0.7, beta=0)

    # RGB -> BGR 변환 최소화
    fused_bgr = cv2.cvtColor(fused_adjusted, cv2.COLOR_RGB2BGR)

    # 6. 디스플레이용 변환(원본 해상도로 리사이즈) 및 화면 출력
    fused_resized = cv2.resize(fused_bgr, (original_w, original_h))
    dehazed_resized = cv2.resize(dehazed_rgb, (original_w, original_h))

    combined = np.hstack((frame_rgb, dehazed_resized, fused_resized))
    cv2.imshow('Original | Dehazed | Fused', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 메모리 관리
torch.cuda.empty_cache()

cap_rgb.release()
cap_ir.release()
cv2.destroyAllWindows()