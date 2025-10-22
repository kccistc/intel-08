# File: main_carla_client.py
import carla
import random
import time
import configparser
import numpy as np

from video_streamer import VideoStreamer
from vehicle_data_sender import VehicleDataSender

# --- 이미지 처리 콜백 함수 ---
def process_image(image, streamers, camera_index):
    """CARLA 센서가 이미지를 생성할 때마다 호출되는 함수."""
    # BGRA -> BGR 변환
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # 해당 카메라의 스트리머에게 프레임 전달
    streamers[camera_index].send_frame(array)

def main():
    # --- 설정 읽기 ---
    config = configparser.ConfigParser()
    config.read('config.ini')

    carla_config = config['CARLA']
    stream_config = config['STREAMING']
    data_config = config['VEHICLE_DATA']
    
    client = None
    vehicle = None
    
    try:
        # --- CARLA 접속 및 차량 생성 ---
        client = carla.Client(carla_config['host'], int(carla_config['port']))
        client.set_timeout(10.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.tesla.*'))
        spawn_point = random.choice(world.get_map().get_random_spawn_point())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle.set_autopilot(True)
        print(f"🚗 Vehicle {vehicle.type_id} spawned.")

        # --- 전문 모듈 초기화 ---
        streamers = [
            VideoStreamer(
                stream_config['target_ip'],
                int(stream_config['base_port']) + i,
                800, 600 # 카메라 해상도
            ) for i in range(int(stream_config['num_cameras']))
        ]
        
        data_sender = VehicleDataSender(
            data_config['serial_port'],
            int(data_config['baud_rate'])
        )

        # --- 카메라 센서 생성 및 콜백 연결 ---
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')

        # 6방향 카메라 위치 설정 (예시)
        spawn_points = [
            carla.Transform(carla.Location(x=2.5, z=0.7)), # Front
            # ... (나머지 5개 카메라 위치) ...
        ]
        
        for i in range(len(spawn_points)):
            camera = world.spawn_actor(camera_bp, spawn_points[i], attach_to=vehicle)
            # lambda를 이용해 각 카메라 인덱스를 콜백 함수에 전달
            camera.listen(lambda image, index=i: process_image(image, streamers, index))

        # --- 메인 루프 ---
        print("✅ System running. Press Ctrl+C to stop.")
        while True:
            # 차량 데이터 전송
            data_sender.send_data(vehicle)
            # 1초 대기
            time.sleep(1)

    finally:
        # --- 종료 처리 ---
        print("\n🛑 Cleaning up actors and connections...")
        if vehicle:
            vehicle.destroy()
        if client:
            # CARLA 0.9.12+ 에서는 client.apply_batch([carla.command.DestroyActor(x) for x in actor_list]) 방식 권장
            pass
        print("Cleanup complete.")

if __name__ == '__main__':
    main()