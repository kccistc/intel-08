#!/usr/bin/env python
# -*- coding: utf-8 -*-

# carla 라이브러리는 이제 필요 없으므로 주석 처리 또는 삭제
# import carla
import requests # 💡 HTTP 요청을 위한 라이브러리
import time
import cv2
import numpy as np
from openvino import Core
from flask import Flask, Response, jsonify
import threading
import os
from datetime import datetime
import json

# --- 설정값 ---
CARLA_BRIDGE_URL = "http://localhost:5001" # 💡 CARLA 브릿지 서버 주소

app = Flask(__name__)
# 원본 프레임은 이제 브릿지에서 받아오므로, 처리된 프레임만 저장
processed_frames = {'cctv1': np.zeros((480, 720, 3), dtype=np.uint8), 'cctv2': np.zeros((480, 720, 3), dtype=np.uint8)}
detection_status = {'cctv1': False, 'cctv2': False}
status_lock = threading.Lock()

# camera_callback 함수는 이제 carla_bridge.py에 있으므로 여기서는 삭제합니다.

class InferencePlayer: # 💡 클래스 이름을 더 명확하게 변경
    def __init__(self):
        self.is_running = True
        self.warning_sent_times = {'cctv1': 0, 'cctv2': 0}
        self.warning_cooldown = 5 # 5초 간격
        
        # 💡 카메라 위치는 하드코딩하거나, 브릿지에서 API로 제공받을 수 있습니다. 여기서는 간단하게 하드코딩.
        self.camera_locations = {
            'cctv1': {'x': 5, 'y': -172.62, 'z': 22.35},
            'cctv2': {'x': -20, 'y': 22, 'z': 24}
        }

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

    # (send_v2x_warning 함수는 변경사항 없음)
    def send_v2x_warning(self, camera_name, frame):
        current_time = time.time()
        if current_time - self.warning_sent_times.get(camera_name, 0) < self.warning_cooldown:
            return
        
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")

        # JSON 이벤트 저장
        MESSAGE_DIR = "./events"
        os.makedirs(MESSAGE_DIR, exist_ok=True)
        json_filename = f"event_{camera_name}_{timestamp_str}.json"
        json_filepath = os.path.join(MESSAGE_DIR, json_filename)

        event_data = {
            "timestamp": now.isoformat(),
            "camera_id": camera_name,
            "event_type": "accident_detected",
            "location": self.camera_locations.get(camera_name, "Unknown")
        }

        try:
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=4, ensure_ascii=False)
            print(f"✅ 이벤트 정보를 '{json_filepath}' 경로에 JSON 파일로 저장했습니다.")
        except Exception as e:
            print(f"❗️[오류] JSON 파일 저장 중 오류 발생: {e}")

        

        self.warning_sent_times[camera_name] = current_time

    # 💡 CARLA 관련 함수(setup_simulation, _setup_sensors, cleanup)는 모두 삭제
    
    def run_inference_on_frame(self, frame, compiled_model, output_layer, camera_name, timestamp_str):
        original_h, original_w = frame.shape[:2]; resized_image = cv2.resize(frame, (self.input_w, self.input_h))
        transposed_image = resized_image.transpose(2, 0, 1); input_tensor = np.expand_dims(transposed_image, axis=0).astype(np.float32) / 255.0
        results = compiled_model([input_tensor])[output_layer]; detections = results[0]
        detection_found = False
        boxes = []
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            if confidence >= self.confidence_threshold:
                if not detection_found: # 첫 감지 시에만 이미지 저장
                    detection_found = True
                    # 프레임 이미지 저장 (바운딩 박스 그리기 전)
                    IMAGE_DIR = f"./accident_{camera_name}"
                    os.makedirs(IMAGE_DIR, exist_ok=True)
                    image_filename = f"frame_{timestamp_str}.jpg"
                    image_filepath = os.path.join(IMAGE_DIR, image_filename)
                    
                    try:
                        cv2.imwrite(image_filepath, frame)
                        print(f"✅ 사고 프레임을 '{image_filepath}' 경로에 이미지 파일로 저장했습니다.")
                    except Exception as e:
                        print(f"❗️[오류] 프레임 이미지 파일 저장 중 오류 발생: {e}")

                x1_s, y1_s = int(x1 * original_w / self.input_w), int(y1 * original_h / self.input_h)
                x2_s, y2_s = int(x2 * original_w / self.input_w), int(y2 * original_h / self.input_h)
                cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 0, 255), 2)
                cv2.putText(frame, f"accident : {confidence:.2f}", (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                boxes.append((x1_s, y1_s, x2_s, y2_s, confidence))
        return frame, detection_found, boxes

    def get_frame_from_bridge(self, camera_name):
        """💡 CARLA 브릿지에서 프레임을 가져오는 함수"""
        try:
            response = requests.get(f"{CARLA_BRIDGE_URL}/get_frame/{camera_name}", timeout=0.5)
            if response.status_code == 200:
                # 응답 받은 바이너리 데이터를 Numpy 배열(이미지)로 디코딩
                np_arr = np.frombuffer(response.content, np.uint8)
                return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except requests.exceptions.RequestException as e:
            # print(f"브릿지 연결 오류: {e}") # 너무 자주 출력될 수 있으므로 주석 처리
            pass
        return None # 실패 시 None 반환

    def run(self):
        """💡 메인 로직: 브릿지에서 프레임을 받아와 추론 실행"""
        print("✅ 추론 플레이어 시작. CARLA 브릿지에서 프레임을 가져옵니다.")
        while self.is_running:
            # 각 카메라에 대해 브릿지에서 프레임 가져오기
            frame1 = self.get_frame_from_bridge('cctv1')
            frame2 = self.get_frame_from_bridge('cctv2')
            
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")

            if frame1 is not None:
                cctv1_processed, detected1, boxes1 = self.run_inference_on_frame(frame1, self.compiled_model1, self.output_layer1, 'cctv1', timestamp_str)
                if detected1: self.send_v2x_warning('cctv1', cctv1_processed)
                with status_lock:
                    detection_status['cctv1'] = detected1
                processed_frames['cctv1'] = cctv1_processed
            
            if frame2 is not None:
                cctv2_processed, detected2, boxes2 = self.run_inference_on_frame(frame2, self.compiled_model2, self.output_layer2, 'cctv2', timestamp_str)
                if detected2: self.send_v2x_warning('cctv2', cctv2_processed)
                with status_lock:
                    detection_status['cctv2'] = detected2
                processed_frames['cctv2'] = cctv2_processed
            
            time.sleep(0.03) # 루프 간격 조절

    def quit(self): 
        self.is_running = False

# (이하 Flask API 및 main 함수는 거의 동일)
def generate_frame(camera_name):
    while True:
        with status_lock:
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
    print("🌐 메인 Flask 서버 시작 - http://localhost:5000"); app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main():
    player = None
    try:
        flask_thread = threading.Thread(target=run_flask, daemon=True); flask_thread.start()
        # 💡 CARLA 클라이언트 연결 부분 삭제, InferencePlayer 생성 및 실행
        player = InferencePlayer()
        player.run()
    except KeyboardInterrupt: 
        print("\n프로그램 종료 요청...")
    finally:
        if player: 
            player.quit()
            # cleanup은 carla_bridge.py에서 처리하므로 여기서는 호출 안 함

if __name__ == '__main__': 
    main()