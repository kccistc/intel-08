import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# === 디헤이징을 위한 간단한 흐림 보정 함수 (예: DCP 간략 버전) ===
def estimate_dark_channel(img, patch_size=15):
    # img: HxWx3, 값 범위 [0,255]
    # dark channel: 각 픽셀 주변 패치 중 채널 최소 픽셀값의 최대값
    min_per_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_per_channel, kernel)
    return dark

def estimate_atmospheric_light(img, dark):
    # 상위 0.1% 밝은 영역의 평균을 A로 설정
    h, w = dark.shape
    num_pixels = h * w
    num_bright = max(num_pixels // 1000, 1)
    # flatten 정렬
    indices = np.argsort(dark.ravel())[::-1][:num_bright]
    img_flat = img.reshape((-1,3))
    atmo = np.mean(img_flat[indices], axis=0)
    return atmo

def estimate_transmission(img, A, omega=0.95, patch_size=15):
    # t(x) = 1 - omega * min ( over c ) ( min over patch ( I_c(x)/A_c ) )
    norm = img / A
    min_per_channel = np.min(norm, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_norm = cv2.erode(min_per_channel, kernel)
    transmission = 1 - omega * dark_norm
    return transmission

def recover_image(img, t, A, t0=0.1):
    # J(x) = (I(x) - A) / t(x) + A
    t = np.clip(t, t0, 1.0)
    J = (img.astype(np.float32) - A) / t[..., np.newaxis] + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

def dehaze_simple(img_bgr):
    img = img_bgr.astype(np.float32)
    dark = estimate_dark_channel(img, patch_size=15)
    A = estimate_atmospheric_light(img, dark)
    t = estimate_transmission(img, A, omega=0.95, patch_size=15)
    J = recover_image(img, t, A, t0=0.1)
    return J

# === YOLO 객체 탐지 부분 (Ultralytics YOLOv5 / YOLOv8 가정) ===
from ultralytics import YOLO  # pip install ultralytics

def detect_objects(image):
    # image: numpy array, BGR
    # YOLO 모델 로드 (사전 훈련된 것)
    model = YOLO("yolov8n.pt")  # 또는 yolov5 모델
    results = model(image)  # 결과 객체
    return results

def draw_detections(image, results):
    # results: ultralytics 결과 객체
    image_out = image.copy()
    for r in results:
        boxes = r.boxes  # .xyxy, .conf, .cls
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cv2.rectangle(image_out, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_out, f"{cls_id}:{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return image_out

# === 전체 파이프라인 ===
def process_frame(frame_bgr):
    dehazed = dehaze_simple(frame_bgr)
    results = detect_objects(dehazed)
    out = draw_detections(dehazed, results)
    return out, dehazed

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_img, dehazed_img = process_frame(frame)
        # 원본, 디헤이징된 이미지, 탐지 결과 영상 보기
        combo = np.hstack([frame, dehazed_img, out_img])
        cv2.imshow("orig | dehazed | detection", combo)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()