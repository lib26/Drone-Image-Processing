import numpy as np
import cv2 as cv

filename = 'house.jpeg'
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
#보통 0.04를 한다.
dst = cv.cornerHarris(gray, 2, 3, 0.04)
# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
# dst 안에 있는 max나 평균값등을 알아봐라.
# 최종은 local maxima 해서 많은 1을 줄이는 거..?
# 1인 좌표만 서클함수? 써서 동그라미로 표시해봐라

# 0.001로 줄였을 때 0.5로
img[dst > 0.01 * dst.max()] = [0, 0, 255]
# dst가 쓰레쉬홀더 값을 넘어가면 이미지 픽셀값을 빨간색으로 바꿔라 GBR 순서임.
cv.imshow('dst', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
