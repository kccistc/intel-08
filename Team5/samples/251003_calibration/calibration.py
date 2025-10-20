import cv2
import numpy as np
import glob
import json

# 체스보드 사이즈 (내부 코너 개수)
pattern_size = (8, 5)  # 예: 8×5
square_size = 4.0  # 실제 체스보드 한 칸 크기 단위 (적당히 설정)

# object points (3D 좌표)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D
imgpoints_rgb = []  # RGB 카메라 이미지 상의 2D 포인트
imgpoints_ir = []   # IR 카메라 이미지 상의 2D 포인트

# 이미지 쌍 읽기
rgb_imgs = sorted(glob.glob('rgb_calib/*.png'))
ir_imgs = sorted(glob.glob('ir_calib/*.png'))

# 첫 번째 이미지 크기 가져오기 (후에 사용할 크기)
img_rgb = cv2.imread(rgb_imgs[0], cv2.IMREAD_GRAYSCALE)
img_ir = cv2.imread(ir_imgs[0], cv2.IMREAD_GRAYSCALE)

for rgb_path, ir_path in zip(rgb_imgs, ir_imgs):
    img_rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    img_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    # 체스보드 코너 찾기
    ret_rgb, corners_rgb = cv2.findChessboardCorners(img_rgb, pattern_size, None)
    ret_ir, corners_ir = cv2.findChessboardCorners(img_ir, pattern_size, None)
    if ret_rgb and ret_ir:
        objpoints.append(objp)
        cv2.cornerSubPix(img_rgb, corners_rgb, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.cornerSubPix(img_ir, corners_ir, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_rgb.append(corners_rgb)
        imgpoints_ir.append(corners_ir)

    # -------------- 디버그용 코너 표시 -------------------
        cv2.drawChessboardCorners(img_rgb, pattern_size, corners_rgb, ret_rgb)
        cv2.drawChessboardCorners(img_ir, pattern_size, corners_ir, ret_ir)
        cv2.imshow('RGB Chessboard', img_rgb)
        cv2.imshow('IR Chessboard', img_ir)
        cv2.waitKey(500)  # 잠시 대기
    else:
        # 코너 찾기 실패 시 로그 출력
        print(f"체스보드 코너를 찾을 수 없습니다: {rgb_path}, {ir_path}")
    # --------------------------------------------------

# 내부 보정
ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints_rgb, img_rgb.shape[::-1], None, None)
ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints_ir, img_ir.shape[::-1], None, None)

# 스테레오 보정
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_rgb, imgpoints_ir,
    mtx1, dist1, mtx2, dist2, img_rgb.shape[::-1],
    criteria=criteria, flags=flags
)

# 영상 정렬 (rectify)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx1, dist1, mtx2, dist2, img_rgb.shape[::-1], R, T
)

#+++++++++++++++++++
print("stereoCalibrate RMS error:", ret_stereo)
print("R1:", R1)
print("P1:", P1)

if ret_stereo > 1.0:  # 기준값은 조정 가능
    print("Stereo calibration error too high. Check images and calibration process.")
    #exit(1)
#++++++++++++++++++++++

size = img_rgb.shape[::-1] # 스테레오 캘리브레이션에 사용된 크기 (W, H)

map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(
    mtx1, dist1, R1, P1, size, cv2.CV_32FC1)
map1_ir, map2_ir = cv2.initUndistortRectifyMap(
    mtx2, dist2, R2, P2, size, cv2.CV_32FC1)

# JSON 저장을 위해 NumPy 배열을 리스트로 변환
calib_params = {
    'map1_rgb': map1_rgb.tolist(),
    'map2_rgb': map2_rgb.tolist(),
    'map1_ir': map1_ir.tolist(),
    'map2_ir': map2_ir.tolist()
}

# 캘리브레이션 파일로 저장
with open('calib_params.json', 'w') as f:
    json.dump(calib_params, f, indent=4)

print("캘리브레이션 맵 데이터가 'calib_params.json' 저장 완료.")