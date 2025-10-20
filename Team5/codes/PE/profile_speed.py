import cv2
import time
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

print("=== 성능 테스트 ===\n")

# 1. 카메라 읽기 속도
cap = cv2.VideoCapture(0)
start = time.time()
for i in range(30):
    ret, frame = cap.read()
elapsed = time.time() - start
print(f"1. 카메라 읽기: {30/elapsed:.1f} FPS")
print(f"   해상도: {frame.shape[1]}x{frame.shape[0]}")

# 2. YOLO 검출 속도
model = YOLO('yolov8n.pt')
start = time.time()
for i in range(10):
    results = model.predict(frame, conf=0.5, classes=[0], verbose=False, imgsz=640)
elapsed = time.time() - start
print(f"2. YOLO (imgsz=640): {10/elapsed:.1f} FPS")

# 3. YOLO (imgsz=416)
start = time.time()
for i in range(10):
    results = model.predict(frame, conf=0.5, classes=[0], verbose=False, imgsz=416)
elapsed = time.time() - start
print(f"3. YOLO (imgsz=416): {10/elapsed:.1f} FPS")

# 4. MoveNet Thunder
interpreter = tf.lite.Interpreter(model_path='movenet_thunder.tflite')
interpreter.allocate_tensors()
start = time.time()
for i in range(10):
    input_image = tf.image.resize_with_pad(tf.convert_to_tensor(frame[:300, :300]), 256, 256)
    input_batch = tf.expand_dims(tf.cast(input_image, dtype=tf.uint8), axis=0)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_batch)
    interpreter.invoke()
elapsed = time.time() - start
print(f"4. MoveNet Thunder: {10/elapsed:.1f} FPS")

# 5. MoveNet Lightning
interpreter2 = tf.lite.Interpreter(model_path='movenet_lightning.tflite')
interpreter2.allocate_tensors()
start = time.time()
for i in range(10):
    input_image = tf.image.resize_with_pad(tf.convert_to_tensor(frame[:300, :300]), 192, 192)
    input_batch = tf.expand_dims(tf.cast(input_image, dtype=tf.uint8), axis=0)
    interpreter2.set_tensor(interpreter2.get_input_details()[0]['index'], input_batch)
    interpreter2.invoke()
elapsed = time.time() - start
print(f"5. MoveNet Lightning: {10/elapsed:.1f} FPS")

cap.release()
