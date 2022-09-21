import numpy as np
import cv2

def brightness(gray, val):
    res_img = gray.copy()

    val += 0.0
    res_img = gray + val
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    return res_img

def contrast(gray, grad, inter):
    res_img = gray.copy() * 0

    # your code here
    res_img = gray*grad+inter
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)
    return res_img

img1 = cv2.imread('SIFT_images/object3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SIFT_images/scene5.jpg', cv2.IMREAD_GRAYSCALE)

img1 = contrast(img1, 1.3, -60)
# create(nfeatrues, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
detector = sift = cv2.xfeatures2d.SIFT_create(sigma=1.0)

keypoints1, descriptor1 = detector.detectAndCompute(img1, None)
keypoints2, descriptor2 = detector.detectAndCompute(img2, None)

img_res_detect1 = cv2.drawKeypoints(img1, keypoints1, None)
img_res_detect2 = cv2.drawKeypoints(img1, keypoints1, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptor1, descriptor2, 2)

ratio_thresh = 0.6
good_matches=[]
for m, n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty( (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8 )
cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img_res_detect1', img_res_detect1)
cv2.imshow('img_res_detect2', img_res_detect1)
cv2.imshow('img_matches', img_matches)

# 객체 추적했으니 객체 영역표시를 해볼거다?

cv2.waitKey(0)