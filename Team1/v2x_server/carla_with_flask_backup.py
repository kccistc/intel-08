#!/usr/bin/env python
# -*- coding: utf-8 -*-

import carla
import random
import time
import cv2
import numpy as np
from openvino import Core
from flask import Flask, Response, jsonify
import threading
import os
from datetime import datetime
import json # 💡 JSON 라이브러리 임포트

# (상단 코드 및 글로벌 변수는 변경사항 없음)
app = Flask(__name__)
camera_frames = {'cctv1': np.zeros((480, 720, 3), dtype=np.uint8), 'cctv2': np.zeros((480, 720, 3), dtype=np.uint8)}
processed_frames = {'cctv1': np.zeros((480, 720, 3), dtype=np.uint8), 'cctv2': np.zeros((480, 720, 3), dtype=np.uint8)}
detection_status = {'cctv1': False, 'cctv2': False}
status_lock = threading.Lock()

def camera_callback(image, data_dict, camera_name):
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_bgra = np.reshape(image_data, (image.height, image.width, 4))
    image_bgr = image_bgra[:, :, :3]
    data_dict[camera_name] = image_bgr

class CarlaPlayer:
    def __init__(self, client):
        self.client = client; self.world = None; self.actor_list = []
        self.is_running = True
        self.warning_sent_times = {'cctv1': 0, 'cctv2': 0}
        self.warning_cooldown = 5 # 5초 간격
        self.camera_locations = {} # 💡 카메라 위치를 저장할 딕셔너리

        # (OpenVINO 모델 초기화 부분은 변경사항 없음)
        self.model1_xml_path = "./model1.xml"; self.model1_bin_path = "./model1.bin"
        self.model2_xml_path = "./model2.xml"; self.model2_bin_path = "./model2.bin"
        self.device = "CPU"; self.confidence_threshold = 0.3; self.input_h, self.input_w = 640, 640
        core = Core()
        model1 = core.read_model(model=self.model1_xml_path, weights=self.model1_bin_path)
        model1.reshape([1, 3, self.input_h, self.input_w])
        self.compiled_model1 = core.compile_model(model=model1, device_name=self.device)
        self.output_layer1 = self.compiled_model1.output(0)
        model2 = core.read_model(model=self.model2_xml_path, weights=self.model2_bin_path)
        model2.reshape([1, 3, self.input_h, self.input_w])
        self.compiled_model2 = core.compile_model(model=model2, device_name=self.device)
        self.output_layer2 = self.compiled_model2.output(0)
        print("✅ OpenVINO 모델 2개 로드 완료.")

    # --- 💡 수정된 V2X 경고 함수 (JSON 저장 방식) ---
    def send_v2x_warning(self, camera_name):
        current_time = time.time()
        if current_time - self.warning_sent_times.get(camera_name, 0) < self.warning_cooldown:
            return
        
        MESSAGE_DIR = "events"
        os.makedirs(MESSAGE_DIR, exist_ok=True)
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"event_{camera_name}_{timestamp_str}.json"
        filepath = os.path.join(MESSAGE_DIR, filename)

        # 저장할 이벤트 데이터 구성
        event_data = {
            "timestamp": now.isoformat(),
            "camera_id": camera_name,
            "event_type": "accident_detected",
            "location": self.camera_locations.get(camera_name, "Unknown")
        }

        try:
            # JSON 파일로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=4, ensure_ascii=False)
            
            self.warning_sent_times[camera_name] = current_time
            print(f"✅ 이벤트 정보를 '{filepath}' 경로에 JSON 파일로 저장했습니다.")
        except Exception as e:
            print(f"❗️[오류] JSON 파일 저장 중 오류 발생: {e}")
    # ---------------------------------------------------
    
    # (setup_simulation 함수는 변경사항 없음)
    def setup_simulation(self):
        self.world = self.client.load_world('Town04'); self.world.wait_for_tick()
        self._setup_sensors(self.world.get_blueprint_library()); print("카메라 설치 및 수신 시작.")
        
    def _setup_sensors(self, bp_lib):
        cctv_bp = bp_lib.find('sensor.camera.rgb'); cctv_bp.set_attribute('image_size_x', '720'); cctv_bp.set_attribute('image_size_y', '480')
        
        cctv1_transform = carla.Transform(carla.Location(x=5, y=-172.62, z=22.35), carla.Rotation(pitch=-57.40, yaw=0.23, roll=0))
        cctv1 = self.world.spawn_actor(cctv_bp, cctv1_transform); cctv1.listen(lambda image: camera_callback(image, camera_frames, 'cctv1')); self.actor_list.append(cctv1)
        # 💡 cctv1 위치 정보 저장
        loc1 = cctv1_transform.location
        self.camera_locations['cctv1'] = {'x': loc1.x, 'y': loc1.y, 'z': loc1.z}

        cctv2_transform = carla.Transform(carla.Location(x=-20, y=22, z=24), carla.Rotation(pitch=-40, yaw=180, roll=0))
        cctv2 = self.world.spawn_actor(cctv_bp, cctv2_transform); cctv2.listen(lambda image: camera_callback(image, camera_frames, 'cctv2')); self.actor_list.append(cctv2)
        # 💡 cctv2 위치 정보 저장
        loc2 = cctv2_transform.location
        self.camera_locations['cctv2'] = {'x': loc2.x, 'y': loc2.y, 'z': loc2.z}
    
    # (run_inference_on_frame, run, quit, cleanup 함수는 변경사항 없음)
    def run_inference_on_frame(self, frame, compiled_model, output_layer):
        original_h, original_w = frame.shape[:2]; resized_image = cv2.resize(frame, (self.input_w, self.input_h))
        transposed_image = resized_image.transpose(2, 0, 1); input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32) / 255.0
        results = compiled_model([input_tensor])[output_layer]; detections = results[0]
        detection_found = False
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            if confidence >= self.confidence_threshold:
                detection_found = True
                x1_s, y1_s = int(x1 * original_w / self.input_w), int(y1 * original_h / self.input_h)
                x2_s, y2_s = int(x2 * original_w / self.input_w), int(y2 * original_h / self.input_h)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 0, 255), 2)
                cv2.putText(frame, f"accident : {confidence:.2f}", (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame, detection_found
    def run(self):
        self.setup_simulation()
        while self.is_running:
            self.world.wait_for_tick()
            cctv1_processed, detected1 = self.run_inference_on_frame(camera_frames['cctv1'].copy(), self.compiled_model1, self.output_layer1)
            cctv2_processed, detected2 = self.run_inference_on_frame(camera_frames['cctv2'].copy(), self.compiled_model2, self.output_layer2)
            if detected1: self.send_v2x_warning('cctv1')
            if detected2: self.send_v2x_warning('cctv2')
            with status_lock:
                detection_status['cctv1'] = detected1
                detection_status['cctv2'] = detected2
            processed_frames['cctv1'] = cctv1_processed; processed_frames['cctv2'] = cctv2_processed
            time.sleep(0.01)
    def quit(self): self.is_running = False
    def cleanup(self):
        print('\n모든 액터를 파괴합니다.'); self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

# (이하 Flask API 및 main 함수는 변경사항 없음)
def generate_frame(camera_name):
    while True:
        frame = processed_frames[camera_name].copy()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)
@app.route('/video_feed/cctv1')
def video_feed_cctv1(): return Response(generate_frame('cctv1'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed/cctv2')
def video_feed_cctv2(): return Response(generate_frame('cctv2'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/health')
def health(): return {'status': 'running'}
@app.route('/api/status')
def api_status():
    with status_lock: return jsonify(detection_status)
def run_flask():
    print("🌐 Flask 서버 시작 - http://localhost:5000"); app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
def main():
    player = None
    try:
        flask_thread = threading.Thread(target=run_flask, daemon=True); flask_thread.start()
        client = carla.Client('localhost', 2000); client.set_timeout(10.0)
        player = CarlaPlayer(client); player.run()
    except KeyboardInterrupt: print("\n프로그램 종료 요청...")
    finally:
        if player: player.quit(); player.cleanup()
if __name__ == '__main__': main()