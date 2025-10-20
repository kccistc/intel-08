import cv2
import torch
from torchvision.transforms import ToTensor, Normalize
import numpy as np
import time

# 모델 로드
model = torch.load('/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/checkpoint/indoor/netG_model_epoch_4.pth', weights_only=False).to('cuda')
model.eval()

# 전처리 설정
transform = ToTensor()
normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

MODEL_INPUT_SIZE = (512, 512)
DISPLAY_WIDTH = 512
DISPLAY_HEIGHT = 512

# 웹캠 설정
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

last_capture_time = time.time()
dehazed_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임의 해상도 저장
    original_h, original_w, _ = frame.shape

    # 모델 입력용 리사이즈
    model_input_frame = cv2.resize(frame, MODEL_INPUT_SIZE)

    current_time = time.time()

    # 3초마다 프레임 처리
    if current_time - last_capture_time >= 3:
        last_capture_time = current_time

        # OpenCV 이미지를 PyTorch 텐서로 변환
        # OpenCV는 BGR 형식이므로 RGB로 변환
        rgb_frame = cv2.cvtColor(model_input_frame, cv2.COLOR_BGR2RGB)
        tensor_image = transform(rgb_frame).unsqueeze(0) # 배치 차원 추가
        input_tensor = normalize(tensor_image).to('cuda')

        with torch.no_grad():
            dehazed_tensor = model(input_tensor)

        # 텐서 결과를 NumPy 배열로 변환하고 OpenCV 형식에 맞게 조정
        dehazed_image = dehazed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        dehazed_image = (dehazed_image * 0.5) + 0.5 # 정규화 되돌리기
        dehazed_image = np.clip(dehazed_image, 0, 1) * 255
        dehazed_image = dehazed_image.astype(np.uint8)
        dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_RGB2BGR)

        # dehazed_image가 초기화되지 않은 상태라면, 원본 프레임을 임시로 사용
        if dehazed_image is None:
            dehazed_image = frame

        # Dehazed 이미지 원본 해상도로 리사이즈
        resized_dehazed = cv2.resize(dehazed_image, (original_w, original_h))

        # 출력할 화면 리사이즈 및 합쳐서 출력
        resized_original = cv2.resize(frame, (original_w, original_h))
        combined = np.hstack((resized_original, resized_dehazed))
        cv2.imshow('Original (Left) vs Dehazed (Right)', combined)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()