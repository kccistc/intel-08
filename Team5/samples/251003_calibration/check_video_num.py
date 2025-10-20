import cv2

MAX_CAMERAS = 5  # 확인할 최대 카메라 장치 번호

print("--- 카메라 장치 번호 확인 시작 ---")

found_cameras = {}

for i in range(MAX_CAMERAS):
    cap = cv2.VideoCapture(i)
    
    # 캡처 객체가 열리고 프레임을 제대로 읽을 수 있는지 확인
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 캡처에 성공하면 정보를 저장
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 카메라 이름 가져오기 (운영체제에 따라 작동하지 않을 수도 있습니다)
            backend_name = cap.getBackendName()
            
            found_cameras[i] = f"해상도: {width}x{height}, 백엔드: {backend_name}"
            
            # 테스트를 위해 미리보기 화면을 잠시 띄워 시각적으로 확인
            cv2.imshow(f'Camera {i}', frame)
            cv2.waitKey(500) # 0.5초 대기 후 닫음
        
        cap.release()
        cv2.destroyAllWindows()
    
print("\n--- 확인 결과 ---")
if found_cameras:
    for index, info in found_cameras.items():
        print(f"장치 번호 {index}: 성공적으로 인식됨 ({info})")
    print("\n테스트 화면을 보고, RGB 카메라와 IR 카메라가 어떤 번호인지 확인.")
else:
    print("인식된 카메라 장치가 없습니다. 연결 상태를 확인하세요.")

print("--------------------")