import cv2
import numpy as np

def enhance_low_light(image):
    # 1) 컬러 이미지를 LAB 색 공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2) L 채널만 추출 (밝기 정보)
    l, a, b = cv2.split(lab)
    
    # 3) CLAHE 적용 (적응형 히스토그램 평활화)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 4) CLAHE 적용된 L 채널과 나머지 채널 합침
    limg = cv2.merge((cl,a,b))
    
    # 5) 다시 BGR 색 공간으로 변환
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def dehaze(image):
    # 간단한 Dehazing: 대기광 추정 후 복원
    # (Dark Channel Prior 기반 간략화 버전)
    
    # 1) 다크 채널 계산
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    dark_channel = cv2.erode(min_channel, kernel)
    
    # 2) 대기광 추정
    A = np.max(dark_channel)
    
    # 3) 투과율 맵 계산 (간단히)
    t = 1 - 0.95 * dark_channel / A
    t = np.clip(t, 0.1, 1)
    
    # 4) 영상 복원
    J = np.empty_like(image, dtype=np.float32)
    for c in range(3):
        J[:,:,c] = (image[:,:,c].astype(np.float32) - A) / t + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    
    return J

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 카메라 인덱스 0번 (라즈베리파이 카메라)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1) 저조도 개선
        enhanced = enhance_low_light(frame)
        
        # 2) Dehazing 적용
        dehazed = dehaze(enhanced)
        
        # 결과 출력
        cv2.imshow('Original', frame)
        cv2.imshow('Enhanced', enhanced)
        cv2.imshow('Dehazed', dehazed)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 종료
            break
    
    cap.release()
    cv2.destroyAllWindows()
