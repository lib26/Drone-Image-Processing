import numpy as np
import cv2

img1 = cv2.imread('SIFT_images/object3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SIFT_images/scene7.jpg', cv2.IMREAD_GRAYSCALE)

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16 ?

# object의 대비효과 contrast과 쓰레시홀드만 조절해도...?

detector = sift = cv2.xfeatures2d.SIFT_create() #create 레퍼런스 참고해봐서 constrast, sigma 값도 바꿔보세요

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# CLAHE 객체에 원본 이미지 입력하여 CLAHE가 적용된 이미지 생성
gray_cont_dst = clahe.apply(img2)

# 얘네 값은 출력해보는 거 추천
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
#clip으로 밝기 조절?

img_res_detect1 = cv2.drawKeypoints(img1, keypoints1, None)
img_res_detect2 = cv2.drawKeypoints(img2, keypoints2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

ratio_thresh = 0.7
good_matches = []
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img_res_detect1', img_res_detect1)
cv2.imshow('img_res_detect2', img_res_detect2)
cv2.imshow('img_matches', img_matches)
cv2.waitKey(0)
