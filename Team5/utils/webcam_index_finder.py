import cv2
import time
import os

# --- 설정 ---
# 테스트할 최대 인덱스 번호 (0부터 시작)
MAX_CAM_INDEX_TO_CHECK = 10 
# 카메라를 초기화할 때 사용할 해상도 (일부 카메라는 특정 해상도가 필요할 수 있음)
WIDTH = 640
HEIGHT = 480
# ----------------

def scan_cameras():
    """
    시스템에 연결된 사용 가능한 카메라 인덱스를 스캔하고 결과를 출력합니다.
    """
    print("---------------------------------------------")
    print(f"[알림] OpenCV 카메라 인덱스 스캔을 시작합니다 (0 ~ {MAX_CAM_INDEX_TO_CHECK - 1}).")
    print("시스템 환경과 설정에 따라 결과가 다를 수 있습니다.")
    print("---------------------------------------------")
    
    available_indices = []

    for index in range(MAX_CAM_INDEX_TO_CHECK):
        print(f"[{index}]번 인덱스 확인 중...", end="", flush=True)
        
        # 1. VideoCapture 객체 생성
        # GStreamer 백엔드를 명시적으로 사용할 수 있습니다 (라즈베리 파이에서 유용).
        # cap = cv2.VideoCapture(index, cv2.CAP_GSTREAMER) 
        cap = cv2.VideoCapture(index)

        # 2. 해상도 설정 시도 (선택 사항이지만 안정성 향상)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        # 3. 카메라가 열렸는지 확인
        if cap.isOpened():
            # 4. 카메라가 실제로 작동하는지 확인하기 위해 프레임 읽기 시도
            ret, frame = cap.read()
            
            if ret:
                # 성공적으로 열리고 프레임도 읽음
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f" -> 성공! (실제 해상도: {actual_w}x{actual_h})")
                available_indices.append(index)
            else:
                # 열렸지만 프레임은 읽을 수 없는 경우 (권한 문제 또는 장치 대기)
                print(" -> 열기 성공, 하지만 프레임 읽기 실패 (다른 프로세스 사용 중일 수 있음)")
        else:
            print(" -> 실패 (카메라 없음 또는 사용 불가능)")
        
        # 5. 다음 인덱스 테스트를 위해 카메라 즉시 해제
        if cap.isOpened():
            cap.release()
            
        time.sleep(0.1) # 시스템 부하를 줄이기 위한 짧은 지연

    print("\n---------------------------------------------")
    if available_indices:
        print(f"[결과] 사용 가능한 카메라 인덱스: {available_indices}")
        print("이 인덱스 번호로 AD, PE 카메라를 지정하십시오.")
    else:
        print("[결과] 사용 가능한 카메라가 감지되지 않았습니다.")
    print("---------------------------------------------")

if __name__ == "__main__":
    # cv2 라이브러리가 설치되어 있는지 확인
    if 'cv2' in globals():
        scan_cameras()
    else:
        print("오류: OpenCV (cv2) 라이브러리가 설치되어 있지 않습니다.")
        print("설치 명령: pip install opencv-python")

