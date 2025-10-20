import cv2
import time
import os
import numpy as np

# --- 설정 ---
CAP_RGB = 0  # RGB 카메라 장치 번호
CAP_IR = 2   # IR 카메라 장치 번호
RGB_DIR = 'rgb_calib'
IR_DIR = 'ir_calib'
COUNT = 0 # 촬영 시작 번호

# 폴더 생성
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(IR_DIR, exist_ok=True)

# 카메라 설정
cap_rgb = cv2.VideoCapture(CAP_RGB)
cap_ir = cv2.VideoCapture(CAP_IR)

if not cap_rgb.isOpened() or not cap_ir.isOpened():
    print("Error: 카메라를 열 수 없습니다. 장치 번호를 확인하세요.")
    exit()

print("체커보드를 화면에 표시하고 's' 키를 눌러 이미지 쌍을 저장하세요. 'q' 키로 종료합니다.")
print("저장 파일명: {00}.png, {01}.png, ...")

while True:
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_ir, frame_ir = cap_ir.read()

    if not ret_rgb or not ret_ir:
        print("경고: 프레임을 읽을 수 없습니다.")
        break

    h_rgb, w_rgb, _ = frame_rgb.shape
    h_ir, w_ir, _ = frame_ir.shape
    
    # 통일할 기준 높이
    target_height = min(h_rgb, h_ir)

    frame_ir_gray = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)
    
    # RGB 프레임 높이 조정
    if h_rgb != target_height:
        # 비율 유지 계산
        target_width_rgb = int(w_rgb * (target_height / h_rgb))
        frame_rgb_resized = cv2.resize(frame_rgb, (target_width_rgb, target_height))
    else:
        frame_rgb_resized = frame_rgb

    # IR 프레임 높이 조정
    if h_ir != target_height:
        target_width_ir = int(w_ir * (target_height / h_ir))
        frame_ir_resized = cv2.resize(frame_ir_gray, (target_width_ir, target_height))
    else:
        frame_ir_resized = frame_ir_gray
    
    frame_ir_display = cv2.cvtColor(frame_ir_resized, cv2.COLOR_GRAY2BGR)

    combined = np.hstack((frame_rgb_resized, frame_ir_display))
    cv2.putText(combined, f'Count: {COUNT}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('RGB (Left) | IR (Right)', combined)

    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('s'):        
        rgb_filename = os.path.join(RGB_DIR, f'{COUNT:02d}.png')
        ir_filename = os.path.join(IR_DIR, f'{COUNT:02d}.png')
        
        # 리사이즈된 프레임을 저장
        success_rgb = cv2.imwrite(rgb_filename, frame_rgb_resized)
        success_ir = cv2.imwrite(ir_filename, frame_ir_resized)

        if success_rgb and success_ir:
            print(f"이미지 쌍 저장 완료: {rgb_filename}, {ir_filename}")
            COUNT += 1
            time.sleep(0.5)
        else:
            print("저장 실패: 파일 쓰기 권한 또는 경로 문제일 수 있습니다.")
            time.sleep(1) # 실패 시 딜레이
            
    elif key == ord('q'):
        break

cap_rgb.release()
cap_ir.release()
cv2.destroyAllWindows()