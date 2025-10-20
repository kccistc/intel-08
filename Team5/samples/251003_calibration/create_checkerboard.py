import cv2
import numpy as np

# 체커보드 사이즈
pattern_size = (9, 6)  # 예: 9x6 (가로 9, 세로 6)
square_size = 50  # 각 사각형 크기

# 체커보드 이미지 생성
board = np.zeros((pattern_size[1] * square_size, pattern_size[0] * square_size), dtype=np.uint8)

# 흑백 사각형 채우기
for i in range(pattern_size[1]):
    for j in range(pattern_size[0]):
        if (i + j) % 2 == 0:
            cv2.rectangle(board, (j * square_size, i * square_size),
                          ((j + 1) * square_size, (i + 1) * square_size), 255, -1)

# 이미지 저장
cv2.imwrite('checkerboard.jpeg', board)

# 이미지 보기
cv2.imshow('Checkerboard', board)
cv2.waitKey(0)
cv2.destroyAllWindows()
