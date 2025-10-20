import cv2
import numpy as np
from openvino.runtime import Core
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import os
import time

# =======================
# 1. 모델 경로 설정
# =======================
det_xml = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Detection.xml"
det_bin = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Detection.bin"

cls_xml = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Classification.xml"
cls_bin = "/home/ubuntu26/workspace/AD_Dataset/Detection_Model/Classification.bin"

for path in [det_xml, det_bin, cls_xml, cls_bin]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {path}")

# =======================
# 2. OpenVINO 모델 로드
# =======================
ie = Core()

det_model = ie.read_model(det_xml, det_bin)
det_compiled = ie.compile_model(det_model, "CPU")
det_input_layer = det_compiled.input(0)
det_output_layer = det_compiled.output(0)

cls_model = ie.read_model(cls_xml, cls_bin)
cls_compiled = ie.compile_model(cls_model, "CPU")
cls_input_layer = cls_compiled.input(0)
cls_output_layer = cls_compiled.output(0)

_, _, cls_h, cls_w = cls_input_layer.shape

# =======================
# 3. 라벨 이름 정의
# =======================
class_names = ["Buoy", "Reef", "Island", "Ship", "Bridge", "Dockside", "Animal"]

# =======================
# 4. NMS 함수 정의
# =======================
def nms(boxes, scores, score_threshold=0.5, iou_threshold=0.5):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices], [scores[i] for i in indices]
    return [], []

# =======================
# 5. IoU 계산 함수
# =======================
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# =======================
# 6. PyTorch Anomaly Detection 모델 로드
# =======================
DEPLOYMENT_FILE = "deployed_obstacle_detector.pt"  # TorchScript 모델
OPTIMAL_THRESHOLD = 0.7
MODEL_INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

try:
    deployed_model = torch.jit.load(DEPLOYMENT_FILE, map_location='cpu')
    deployed_model.eval()
    print(f"✅ 배포 모델 '{DEPLOYMENT_FILE}' 로드 완료.")
except Exception as e:
    print(f"❌ 오류: 배포 파일 로드 실패. {e}")
    exit()

inference_transforms = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# =======================
# 7. 웹캠 열기
# =======================
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ 웹캠 (인덱스 {CAMERA_INDEX})을 열 수 없습니다.")
    exit()
print("✅ 웹캠 열기 성공. 'q'를 눌러 종료합니다.")

last_frame_boxes = []
frame_count = 0
start_time = time.time()

# =======================
# 8. 실시간 루프
# =======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame, _ = frame.shape

    # --------------------------
    # OpenVINO Detection
    # --------------------------
    resized = cv2.resize(frame, (640, 640))
    input_image = np.expand_dims(resized.transpose(2,0,1), 0).astype(np.float32)
    det_results = det_compiled([input_image])[det_output_layer][0]

    boxes, scores = [], []
    for det in det_results:
        x_min, y_min, x_max, y_max, conf = det
        if conf > 0.5:
            x_min = int(x_min / 640 * w_frame)
            y_min = int(y_min / 640 * h_frame)
            x_max = int(x_max / 640 * w_frame)
            y_max = int(y_max / 640 * h_frame)
            boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            scores.append(float(conf))

    filtered_boxes, filtered_scores = nms(boxes, scores)

    # --------------------------
    # Frame smoothing
    # --------------------------
    smoothed_boxes = []
    for box in filtered_boxes:
        matched = False
        for prev_box in last_frame_boxes:
            if iou(box, prev_box) > 0.7:
                smoothed_boxes.append(prev_box)
                matched = True
                break
        if not matched:
            smoothed_boxes.append(box)
    last_frame_boxes = smoothed_boxes.copy()

    # --------------------------
    # Classification + Visualization
    # --------------------------
    for (x, y, w, h) in smoothed_boxes:
        crop = frame[max(0,y-5):min(h_frame,y+h+5), max(0,x-5):min(w_frame,x+w+5)]
        if crop.size == 0:
            continue
        cls_resized = cv2.resize(crop, (cls_w, cls_h))
        cls_resized = cv2.cvtColor(cls_resized, cv2.COLOR_BGR2RGB)
        cls_input = np.expand_dims(cls_resized.transpose(2,0,1), 0).astype(np.float32) / 255.0

        cls_result = cls_compiled([cls_input])[cls_output_layer]
        class_id = int(np.argmax(cls_result))
        score_cls = float(np.max(cls_result))
        label_name = class_names[class_id]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label_name} {score_cls:.2f}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # --------------------------
    # PyTorch Anomaly Detection
    # --------------------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    input_tensor = inference_transforms(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = deployed_model(input_tensor)
    prob = torch.sigmoid(output).item()
    status = "ANOMALY: OBSTACLE" if prob >= OPTIMAL_THRESHOLD else "NORMAL"
    color = (0,0,255) if prob >= OPTIMAL_THRESHOLD else (0,255,0)

    frame_count += 1
    fps = frame_count / (time.time() - start_time)

    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Prob: {prob:.4f}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Detection + Classification + Anomaly", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 프로그램 종료.")
