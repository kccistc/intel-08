import cv2
import glob

# --- 1. 환경 설정: 이제 이 값이 AD, PE 카메라를 식별하는 고유 ID가 됩니다. ---
# 찾으신 AD 카메라의 VID:PID (Vendor ID:Product ID)
AD_CAM_VID_PID = "1b3f:4167"
# 찾으신 PE 카메라의 VID:PID (Vendor ID:Product ID)
PE_CAM_VID_PID = "1908:2310"
# -------------------------------------------------------------------

def get_v4l2_info(device_path):
    """
    주어진 /dev/videoX 경로에서 VID:PID 정보를 추출합니다.
    이 함수는 리눅스 shell 명령어를 호출하여 VID:PID를 찾습니다.
    """
    try:
        # udevadm 명령어를 실행하여 VID와 PID를 파싱합니다.
        # 실행 결과에서 idVendor와 idProduct 값을 추출합니다.
        import subprocess
        
        # udevadm 실행 및 출력 캡처
        result = subprocess.run(
            ['udevadm', 'info', '--name', device_path, '--attribute-walk'],
            capture_output=True, text=True, check=True, timeout=5
        )
        
        output = result.stdout
        
        vid = None
        pid = None
        
        # 출력에서 idVendor와 idProduct를 찾습니다. (가장 먼저 나오는 값 사용)
        for line in output.splitlines():
            if 'ATTRS{idVendor}' in line and vid is None:
                # ATTRS{idVendor}=="xxxx" 에서 xxxx 값만 추출
                vid = line.split('"')[1]
            if 'ATTRS{idProduct}' in line and pid is None:
                # ATTRS{idProduct}=="xxxx" 에서 xxxx 값만 추출
                pid = line.split('"')[1]
            
            if vid and pid:
                break
        
        if vid and pid:
            return f"{vid}:{pid}"
        else:
            print(f"Error: Could not find VID/PID for {device_path} in udevadm output.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running udevadm for {device_path}: {e}")
        return None
    except FileNotFoundError:
        print("Error: udevadm command not found. Ensure it is installed and in PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {device_path}: {e}")
        return None

def find_camera_by_vid_pid():
    """
    모든 /dev/video 장치를 스캔하여 설정된 VID:PID에 맞는 인덱스를 찾습니다.
    """
    ad_index = -1
    pe_index = -1
    
    # /dev/video* 형식의 모든 장치 파일을 찾습니다.
    video_devices = sorted(glob.glob('/dev/video*'))
    
    if not video_devices:
        print("경고: 시스템에서 /dev/video 장치를 찾을 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
        return ad_index, pe_index

    print(f"총 {len(video_devices)}개의 비디오 장치 발견. 고유 ID 확인 중...")

    for device_path in video_devices:
        # 장치 경로에서 인덱스 번호를 추출합니다 (예: /dev/video0 -> 0)
        try:
            cam_index = int(device_path.replace('/dev/video', ''))
        except ValueError:
            continue # video 뒤에 숫자가 아닌 다른 문자가 있다면 건너뜁니다.

        # 해당 장치의 VID:PID 정보를 얻습니다.
        vid_pid = get_v4l2_info(device_path)
        
        if vid_pid is None:
            print(f"  > {device_path} (인덱스 {cam_index}): ID 확인 실패.")
            continue

        print(f"  > {device_path} (인덱스 {cam_index}): 고유 ID = {vid_pid}")

        if vid_pid == AD_CAM_VID_PID and ad_index == -1:
            ad_index = cam_index
            print(f"    -> AD 카메라 매칭 완료!")
        elif vid_pid == PE_CAM_VID_PID and pe_index == -1:
            pe_index = cam_index
            print(f"    -> PE 카메라 매칭 완료!")

        if ad_index != -1 and pe_index != -1:
            break # 두 카메라를 모두 찾으면 검색을 중지합니다.

    return ad_index, pe_index

def initialize_and_test_cameras(ad_index, pe_index):
    """
    찾은 인덱스로 카메라를 초기화하고 간단한 프레임 캡처 테스트를 수행합니다.
    """
    print("\n--- 카메라 초기화 및 테스트 시작 ---")
    
    # 1. AD 카메라 초기화
    if ad_index != -1:
        print(f"AD 카메라 (인덱스 {ad_index}) 연결 시도...")
        ad_cap = cv2.VideoCapture(ad_index)
        if not ad_cap.isOpened():
            print(f"오류: AD 카메라 (인덱스 {ad_index})를 열 수 없습니다.")
            return

        # 테스트 프레임 읽기
        ret_ad, frame_ad = ad_cap.read()
        ad_cap.release() # 테스트 후 해제
        
        if ret_ad:
            print(f"성공: AD 카메라 연결 확인 완료. 해상도: {frame_ad.shape[1]}x{frame_ad.shape[0]}")
        else:
            print("오류: AD 카메라에서 프레임을 읽을 수 없습니다.")
    else:
        print("AD 카메라를 찾지 못했습니다.")

    # 2. PE 카메라 초기화
    if pe_index != -1:
        print(f"PE 카메라 (인덱스 {pe_index}) 연결 시도...")
        pe_cap = cv2.VideoCapture(pe_index)
        if not pe_cap.isOpened():
            print(f"오류: PE 카메라 (인덱스 {pe_index})를 열 수 없습니다.")
            return

        # 테스트 프레임 읽기
        ret_pe, frame_pe = pe_cap.read()
        pe_cap.release() # 테스트 후 해제

        if ret_pe:
            print(f"성공: PE 카메라 연결 확인 완료. 해상도: {frame_pe.shape[1]}x{frame_pe.shape[0]}")
        else:
            print("오류: PE 카메라에서 프레임을 읽을 수 없습니다.")
    else:
        print("PE 카메라를 찾지 못했습니다.")

    print("--- 카메라 초기화 및 테스트 완료 ---")


if __name__ == "__main__":
    ad_index, pe_index = find_camera_by_vid_pid()
    
    if ad_index != -1 or pe_index != -1:
        print(f"\n최종 결과: AD 카메라 인덱스: {ad_index}, PE 카메라 인덱스: {pe_index}")
        initialize_and_test_cameras(ad_index, pe_index)
    else:
        print("\n최종 결과: AD/PE 카메라 중 어느 것도 설정된 고유 ID로 찾을 수 없습니다. 연결 상태를 확인하거나 VID:PID 값을 재확인하세요.")

