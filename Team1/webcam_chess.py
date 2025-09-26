import cv2
import torch
import numpy as np
import random
import time

def find_board_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(largest_contour) < 5000: 
        return None

    perimeter = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None

def get_perspective_transform(corners, width=800, height=800):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
        
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return matrix

def convert_pixel_to_chess_coord(pixel_coord, matrix, board_size=800):
    px = np.array([[pixel_coord]], dtype="float32")
    transformed_px = cv2.perspectiveTransform(px, matrix)
    
    if transformed_px is None:
        return None
        
    x, y = transformed_px[0][0]
    
    if not (0 <= x < board_size and 0 <= y < board_size):
        return None
        
    square_size = board_size / 8
    col = int(x // square_size)
    row = 7 - int(y // square_size)
    
    files = "abcdefgh"
    ranks = "12345678"
    
    return f"{files[col]}{ranks[row]}"

PROCESS_EVERY_N_FRAMES = 5  # 5프레임마다 한 번씩만 연산
RESIZE_WIDTH = 640          # 처리할 이미지 너비 (클수록 정확하지만 느려짐)
CONFIDENCE_THRESHOLD = 0.6  # 최소 신뢰도

model_path = './runs/train/exp3/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.names]

# 2. 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

frame_count = 0
last_detections = []
last_matrix = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # ✨ N 프레임마다 한 번씩만 연산 수행
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # 원본 프레임 비율 계산
        h, w, _ = frame.shape
        scale = RESIZE_WIDTH / w
        
        # 작은 이미지로 리사이즈하여 처리 속도 향상
        small_frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
        
        corners = find_board_corners(small_frame)
        
        if corners is not None:
            # 리사이즈된 이미지의 코너 좌표를 원본 이미지 좌표로 변환
            original_corners = corners / scale
            last_matrix = get_perspective_transform(original_corners)
            
            results = model(small_frame)
            df = results.pandas().xyxy[0]
            
            # 현재 탐지 결과를 저장
            last_detections = []
            for i in range(len(df)):
                if df.iloc[i]['confidence'] > CONFIDENCE_THRESHOLD:
                    # 모든 좌표를 원본 크기로 다시 스케일업
                    xmin = int(df.iloc[i]['xmin'] / scale)
                    ymin = int(df.iloc[i]['ymin'] / scale)
                    xmax = int(df.iloc[i]['xmax'] / scale)
                    ymax = int(df.iloc[i]['ymax'] / scale)
                    name = df.iloc[i]['name']
                    class_id = int(df.iloc[i]['class'])
                    last_detections.append((xmin, ymin, xmax, ymax, name, class_id))

    if last_matrix is not None:
        for det in last_detections:
            xmin, ymin, xmax, ymax, name, class_id = det
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

            chess_coord = convert_pixel_to_chess_coord((center_x, center_y), last_matrix)
            
            color = colors[class_id]
            
            if chess_coord:
                label = f"{name}: {chess_coord}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Real-time Chess Detection (Optimized)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()