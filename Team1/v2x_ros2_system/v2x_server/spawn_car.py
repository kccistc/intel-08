import carla
import random
import time

def main():
    actor_list = []

    try:
        # --- 1. CARLA 서버에 연결 ---
        client = carla.Client('localhost', 2000)
        # 타임아웃을 10초로 넉넉하게 설정
        client.set_timeout(10.0) 
        world = client.get_world()

        print("CARLA 서버에 성공적으로 연결되었습니다.")
        
        # --- 2. 사용 가능한 차량 블루프린트 및 스폰 지점 가져오기 ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        
        spawn_points = world.get_map().get_spawn_points()
        
        number_of_vehicles = 150
        
        if len(spawn_points) < number_of_vehicles:
            print(f"경고: 맵에 {len(spawn_points)}개의 스폰 지점만 존재합니다. {len(spawn_points)}대의 차량만 스폰합니다.")
            number_of_vehicles = len(spawn_points)

        print(f"{number_of_vehicles}대의 차량을 랜덤하게 스폰합니다...")

        # --- 3. 차량 스폰 및 자율주행 설정 (2단계 배치 처리) ---
        
        # 1단계: 차량 스폰 명령 배치 만들기
        batch_spawn = []
        for _ in range(number_of_vehicles):
            vehicle_bp = random.choice(vehicle_bps)
            # 스폰 지점이 겹치지 않도록 pop 사용
            if spawn_points:
                spawn_point = spawn_points.pop(random.randint(0, len(spawn_points) - 1))
                command = carla.command.SpawnActor(vehicle_bp, spawn_point)
                batch_spawn.append(command)
        
        # 스폰 배치 실행 및 결과 확인
        responses = client.apply_batch_sync(batch_spawn, True)
        
        # 2단계: 성공적으로 스폰된 차량에 대한 자율주행 설정 배치 만들기
        batch_autopilot = []
        for response in responses:
            if response.has_error():
                print(f"  - 차량 스폰 실패: {response.error}")
            else:
                # 성공적으로 스폰된 액터를 리스트에 추가 (나중에 정리하기 위함)
                vehicle = world.get_actor(response.actor_id)
                actor_list.append(vehicle)
                # 자율주행 설정 명령을 배치에 추가
                command = carla.command.SetAutopilot(response.actor_id, True)
                batch_autopilot.append(command)
        
        # 자율주행 설정 배치 실행
        client.apply_batch_sync(batch_autopilot)

        print(f"\n총 {len(actor_list)}대의 차량이 성공적으로 스폰되었고, 자율주행을 시작합니다.")
        print("이 스크립트를 종료하려면 터미널에서 Ctrl+C 를 누르세요.")

        # 스크립트가 바로 종료되지 않고 계속 실행되도록 유지
        while True:
            time.sleep(1)

    finally:
        # --- 4. 스크립트 종료 시 스폰했던 모든 차량 제거 ---
        if actor_list:
            print('\n스폰했던 모든 차량을 파괴하여 시뮬레이션을 정리합니다.')
            destroy_batch = [carla.command.DestroyActor(actor) for actor in actor_list]
            client.apply_batch_sync(destroy_batch)
            print('정리 완료.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - 사용자에 의해 종료되었습니다.')
    except Exception as e:
        print(f"오류 발생: {e}")