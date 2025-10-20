#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 이 스크립트는 Python 3.7 및 CARLA 0.9.13 환경에서 실행되어야 합니다.
import carla
import numpy as np
import cv2
from flask import Flask, Response
import threading
import time

app = Flask(__name__)
# CARLA 센서로부터 받은 원본 프레임을 저장할 글로벌 변수
raw_camera_frames = {
    'cctv1': np.zeros((480, 720, 3), dtype=np.uint8),
    'cctv2': np.zeros((480, 720, 3), dtype=np.uint8)
}
frame_lock = threading.Lock()

def camera_callback(image, camera_name):
    """CARLA 센서 콜백 함수. 프레임을 Numpy 배열로 변환하여 저장합니다."""
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_bgra = np.reshape(image_data, (image.height, image.width, 4))
    image_bgr = image_bgra[:, :, :3]
    with frame_lock:
        raw_camera_frames[camera_name] = image_bgr

class CarlaBridge:
    def __init__(self, client):
        self.client = client
        self.world = None
        self.actor_list = []
        self.is_running = True

    def setup_simulation(self):
        """CARLA 0.9.13 버전에 맞춰 시뮬레이션을 설정합니다."""
        # 0.9.13에서는 get_world()를 사용하고, 필요시 맵을 로드합니다.
        current_map = self.client.get_world().get_map().name.split('/')[-1]
        if current_map != 'Town04':
            print(f"현재 맵 '{current_map}'에서 'Town04'로 변경합니다...")
            self.client.load_world('Town04')
        
        self.world = self.client.get_world()
        self.world.wait_for_tick()
        self._setup_sensors(self.world.get_blueprint_library())
        print("✅ CARLA 브릿지: 카메라 설치 및 프레임 수신 시작.")

    def _setup_sensors(self, bp_lib):
        cctv_bp = bp_lib.find('sensor.camera.rgb')
        cctv_bp.set_attribute('image_size_x', '720')
        cctv_bp.set_attribute('image_size_y', '480')
        
        # CCTV 1
        cctv1_transform = carla.Transform(carla.Location(x=5, y=-172.62, z=22.35), carla.Rotation(pitch=-57.40, yaw=0.23, roll=0))
        cctv1 = self.world.spawn_actor(cctv_bp, cctv1_transform)
        cctv1.listen(lambda image: camera_callback(image, 'cctv1'))
        self.actor_list.append(cctv1)

        # CCTV 2
        cctv2_transform = carla.Transform(carla.Location(x=-20, y=22, z=24), carla.Rotation(pitch=-40, yaw=180, roll=0))
        cctv2 = self.world.spawn_actor(cctv_bp, cctv2_transform)
        cctv2.listen(lambda image: camera_callback(image, 'cctv2'))
        self.actor_list.append(cctv2)

    def run(self):
        """메인 루프. 시뮬레이션 틱을 유지합니다."""
        self.setup_simulation()
        while self.is_running:
            self.world.wait_for_tick()
            time.sleep(0.01) # CPU 사용량 조절

    def quit(self):
        self.is_running = False

    def cleanup(self):
        print('\nCARLA 브릿지: 모든 액터를 파괴합니다.')
        if self.client and self.actor_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

# --- Flask API 엔드포인트 ---
@app.route('/get_frame/<camera_name>')
def get_frame(camera_name):
    """지정된 카메라의 최신 프레임을 JPEG 형식으로 반환합니다."""
    if camera_name not in raw_camera_frames:
        return "Camera not found", 404
    
    with frame_lock:
        frame = raw_camera_frames[camera_name].copy()
    
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    return Response(frame_bytes, mimetype='image/jpeg')

def run_flask():
    # CARLA 통신용 브릿지 서버는 다른 포트(예: 5001)를 사용합니다.
    print("🌐 CARLA 브릿지 서버 시작 - http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)

def main():
    bridge = None
    try:
        # Flask 서버를 별도 스레드에서 실행
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # CARLA 클라이언트 연결 및 메인 루프 실행
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        bridge = CarlaBridge(client)
        bridge.run()
        
    except KeyboardInterrupt:
        print("\n프로그램 종료 요청...")
    finally:
        if bridge:
            bridge.quit()
            bridge.cleanup()

if __name__ == '__main__':
    main()