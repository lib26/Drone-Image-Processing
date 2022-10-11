import numpy as np
import cv2 as cv

# 참고자료 : https://webnautes.tistory.com/1415
from cv2.mat_wrapper import Mat

FLANN_INDEX_LSH = 6

def matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2):
    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)  # 2

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) >= 4:

        keyPoints1 = np.float32([keyPoints1[i] for (_, i) in matches])
        keyPoints2 = np.float32([keyPoints2[i] for (i, _) in matches])

        H, status = cv.findHomography(keyPoints1, keyPoints2, cv.RANSAC, 4.0)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    return matches, H, status


def main(img1, img2):

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    detector = cv.BRISK_create()
    keyPoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keyPoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    print('img1 - %d features, img2 - %d features' % (len(keyPoints1), len(keyPoints2)))

    keyPoints1 = np.float32([keypoint.pt for keypoint in keyPoints1])
    keyPoints2 = np.float32([keypoint.pt for keypoint in keyPoints2])

    matches, H, status = matchKeypoints(keyPoints1, keyPoints2, descriptors1, descriptors2)

    result = cv.warpPerspective(img1, H,
                                (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result


if __name__ == '__main__':
    img1 = cv.imread('panorama/set1/08.png')
    img2 = cv.imread('panorama/set1/07.png')
    img3 = cv.imread('panorama/set1/06.png')

    result = main(img1, img2)
    result2 = main(result, img3)
    print(result2.shape)

    # 이미지 Straightening
    # 참고자료 : https://stackoverflow.com/questions/41995916/opencv-straighten-an-image-with-python
    # ---- 4 corner points of the bounding box
    pts_src = np.array([[0.0, 0.0], [1090.0, 0.0], [0.0, 800.0], [1250.0, 800.0]])

    # ---- 4 corner points of the black image you want to impose it on
    pts_dst = np.array([[0.0, 0.0], [1100.0, 0.0], [0.0, 800.0], [1200.0, 800.0]])

    # ---- forming the black image of specific size
    im_dst = np.zeros((800, 1150, 3), np.uint8)

    # ---- Framing the homography matrix
    h, status = cv.findHomography(pts_src, pts_dst)

    # ---- transforming the image bound in the rectangle to straighten
    im_out = cv.warpPerspective(result2, h, (im_dst.shape[1], im_dst.shape[0]))

    cv.imshow('result2', im_out)
    cv.waitKey()
    cv.destroyAllWindows()