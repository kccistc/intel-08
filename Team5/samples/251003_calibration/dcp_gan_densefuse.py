import sys

sys.path.append('./DCP_GAN_dehazing')
sys.path.append('./densefuse_pytorch')

import cv2
import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
import json

# 사용자 정의 모델 및 유틸리티 임포트
from DCP_GAN_dehazing.dcp_gan import DCP_GAN
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

# --- 모델 로드 및 전처리 설정 ---
dcp_model_path = './DCP_GAN_dehazing/checkpoint/netG_model_epoch_9.pth'
dcp_model = DCP_GAN(input_channels=3, output_channel=3).to('cuda')
checkpoint_dcp = torch.load(dcp_model_path, map_location='cuda')

if isinstance(checkpoint_dcp, dict) and 'model_state_dict' in checkpoint_dcp:
    dcp_model.load_state_dict(checkpoint_dcp['model_state_dict'])
elif isinstance(checkpoint_dcp, dict):
    dcp_model.load_state_dict(checkpoint_dcp)
else:
    dcp_model = checkpoint_dcp.to('cuda')

dcp_model.eval()

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

dehazing_normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
to_tensor = ToTensor()

MODEL_INPUT_SIZE = (512, 512)

# --- 캘리브레이션 파라미터 로드 ---
calib_file = 'calib_params.json' 
map1_rgb, map2_rgb, map1_ir, map2_ir = load_calibration_params(calib_file)

# map1_ir = map1_ir.astype(np.int32)
# map2_ir = map2_ir.astype(np.int32)

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

    # 2. DCP-GAN 안개 제거 (RGB)
    rgb_input = cv2.resize(frame_rgb, MODEL_INPUT_SIZE)
    rgb_input_rgb = cv2.cvtColor(rgb_input, cv2.COLOR_BGR2RGB)
    tensor_rgb = to_tensor(rgb_input_rgb).unsqueeze(0).to('cuda')
    norm_rgb = dehazing_normalize(tensor_rgb)

    with torch.no_grad():
        dehazed_tensor = dcp_model(norm_rgb)
		
	# 3. DCP-GAN 출력(0~1)과 IR 영상(0~1) 텐서 준비
    dehazed_output = (dehazed_tensor.squeeze(0) * 0.5) + 0.5  # [0,1]

    # IR 영상 흑백 변환 및 텐서화 후 정규화
    if aligned_ir_resized.ndim == 3:
        ir_gray = cv2.cvtColor(aligned_ir_resized, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = aligned_ir_resized
    ir_input = to_tensor(ir_gray).to('cuda')
    
    # 4. DenseFuse 실행
    with torch.no_grad():

        features_rgb = dense_model.encoder(dehazed_output.unsqueeze(0))[0] # 결과: [1, 64, H, W]
        ir_input_3ch = ir_input.repeat(3, 1, 1)
        features_ir = dense_model.encoder(ir_input_3ch.unsqueeze(0))[0]        # 결과: [1, 64, H, W]

        fused_features_list = dense_model.fusion([features_rgb], [features_ir], strategy_type='addition')
        input_decoder = fused_features_list[0] # 결과: [1, 64, H, W]

        fused_tensor = dense_model(input_decoder)
		
		# 5. 결과 후처리
    fused_np = fused_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    fused_uint8 = np.clip(fused_np * 255, 0, 255).astype(np.uint8)

    # 밝기/대비 조절
    alpha_control = 0.7  # 0.5 ~ 1.0 사이에서 최적의 값을 찾으세요. (낮을수록 어두워짐)
    beta_control = 0

    fused_adjusted = cv2.convertScaleAbs(fused_uint8, alpha=alpha_control, beta=beta_control)
    
    if fused_adjusted.ndim == 2 or fused_adjusted.shape[2] == 1:
        fused_bgr = cv2.cvtColor(fused_adjusted, cv2.COLOR_GRAY2BGR)
    else:
        fused_bgr = cv2.cvtColor(fused_adjusted, cv2.COLOR_RGB2BGR)

    # 6. 디스플레이용 변환(원본 해상도로 리사이즈) 및 화면 출력
    dehazed_bgr = cv2.cvtColor(
        (dehazed_output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    dehazed_resized = cv2.resize(dehazed_bgr, (original_w, original_h))
    fused_resized = cv2.resize(fused_bgr, (original_w, original_h))

    combined = np.hstack((frame_rgb, dehazed_resized, fused_resized))
    cv2.imshow('RGB (Left) | Dehazed (Middle) | Fused (Right)', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # 메모리 관리를 위한 코드(메모리 누수가 크지 않다면 삭제해도 됨)    
    torch.cuda.empty_cache()

cap_rgb.release()
cap_ir.release()
cv2.destroyAllWindows()