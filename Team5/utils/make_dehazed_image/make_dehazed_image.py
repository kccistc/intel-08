import cv2
import numpy as np
import os
from glob import glob

img_path = './hazy_img/'
save_path = './dehazed_result/'

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
    png_imgs = glob(os.path.join(img_path, '*.png'))
    jpg_imgs = glob(os.path.join(img_path, '*.jpg'))
    hazy_imgs = png_imgs + jpg_imgs

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img_path in hazy_imgs:
        img = cv2.imread(img_path)
    
        # 1) 저조도 개선
        enhanced = enhance_low_light(img)
        
        # 2) Dehazing 적용
        dehazed = dehaze(enhanced)
        
        # 결과 저장
        filename = os.path.basename(img_path)
        save_filepath = os.path.join(save_path, filename)
        cv2.imwrite(save_filepath, dehazed)

    cv2.destroyAllWindows()
