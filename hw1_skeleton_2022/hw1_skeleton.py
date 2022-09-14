#
# Drone and Robotics Homework #1
#
# I am 임인범 201835509
#

import numpy as np
import cv2

# 얘같은 경우 for문 안쓰고 할 수 있는데 넘파이의 강점?
#[열, 행, 넘버] 인데 넘버는 채널이다. 즉 0,1,2 skdha
def convertRGBtoGray(rgb):
    gray  = rgb[:,:,0] * 0.114 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.299
    # 그레
    gray = np.squeeze(gray).astype(np.uint8)
    return gray # 그 차원에 해당하는 메트릭스를 반환. 이때부턴 1차원임 8비트인 뭐...

# val 더해주는 값이고 gray는 바로 위에 변수래. 0으로 초기화된 이미지 만들고
def brightness(gray, val):
    res_img = gray.copy() * 0

    # your code here

    return res_img

def contrast(gray, grad, inter):
    res_img = gray.copy() * 0

    # your code here

    return res_img

def scaling1(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype)

    # 요런 코드 고대로 쓸거래
    # your code here (uncomment below)
    #  forward warping
    # for r in range(h):
    #     for c in range(w):

    return res_img

def scaling2(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype)

    # your code here (uncomment below)
    #  backward warping
    # for r in range(h*s):
    #     for c in range(w*s):

    return res_img

# deg: angle
def rotation(gray, angle):
    res_img = gray.copy() * 0

    #  your code here

    return res_img



if __name__ == '__main__':

    # open image
    img_rgb = cv2.imread('image.png', cv2.IMREAD_COLOR)

    # get dimension 쉐입속성은  3개의 튜플을 전달하는거. 하이트 위드스임. y좌표 기준이라 하이트부터 나옴. 채널은 그레이채널이면 1이 나오고 rgb니까 3채널 나옴
    h, w, ch = img_rgb.shape

    # if you want to know the dimension of img_rgb, remove comment below
    # print(img_rgb.shape)

    # mission 1 : convert color image to grayscale 원본 집어넣고
    img_gray = convertRGBtoGray(img_rgb)

    # mission 2: decrease brightness
    # caution: clip values between 0 ~ 255
    img_bright = brightness(img_gray, 50)

    # mission 3: decrease brightness
    # a: gradient, b:an intercept of y axis
    img_contrast = contrast(img_gray, 1.5, -50)

    # mission 4: scaling
    # move source pixels to target
    img_scaling1 = scaling1(img_gray, 3)

    # mission 5: scaling2
    # move source pixels to target
    img_scaling2 = scaling2(img_gray, 3)

    # mission 6: rotation
    # caution: Rotate the image around the center of the image.
    img_rotation = rotation(img_gray, 30)

    #concatenate results
    img_res1 = cv2.hconcat([img_gray, img_bright, img_contrast])
    img_res2 = cv2.hconcat([img_scaling1, img_scaling2])


    # display input image & results
    cv2.imshow('input image', img_rgb)
    cv2.imshow('gray, bright, contrast', img_res1)
    cv2.imshow('scaling', img_res2)
    cv2.imshow('rotation', img_rotation)
    cv2.waitKey(0)