import numpy as np
import cv2

img1 = cv2.imread('video/track1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('video/track2.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.SIFT_create()

keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

img_res_detect1 = cv2.drawKeypoints(img1, keypoints1, None)
img_res_detect2 = cv2.drawKeypoints(img2, keypoints2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

ratio_thresh = 0.5
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)

# draw motion vector
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for i in range(len(good_matches)):
    cv2.circle(img2_rgb, (int(keypoints2[good_matches[i].trainIdx].pt[0]), int(keypoints2[good_matches[i].trainIdx].pt[1])), 3, (255,0,0), 2)
    cv2.line(img2_rgb, (int(keypoints1[good_matches[i].queryIdx].pt[0]), int(keypoints1[good_matches[i].queryIdx].pt[1])), \
        (int(keypoints2[good_matches[i].trainIdx].pt[0]), int(keypoints2[good_matches[i].trainIdx].pt[1])), (0,255,255),2)

img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('img2', img2_rgb)
cv2.imshow('img_matches', img_matches)
cv2.waitKey(0)