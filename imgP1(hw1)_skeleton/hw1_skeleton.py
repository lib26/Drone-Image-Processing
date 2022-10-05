#
# Drone and Robotics Homework #1
#
# I am 임인범 201835509
#

import numpy as np
import cv2 as cv
import math

# for loop 안 쓰고 할 수 있다. 넘파이 강점
#[열, 행, 넘버] 인데 넘버는 채널. (320, 400, 3)
def convertRGBtoGray(rgb):
    gray  = rgb[:,:,0] * 0.114 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.299
    gray = np.squeeze(gray).astype(np.uint8) # 차원 낮추기
    print(gray.shape)
    return gray # 그 차원에 해당하는 메트릭스를 반환. gray니까 1차원. 8비트

# val 더해주는 값. 0으로 초기화된 이미지를 먼저 만든다.
def brightness(gray, val):
    # res_img = gray.copy() * 0
    # your code here
    res_img = np.clip(gray.copy() + val.__float__(), 0, 255).astype(np.uint8)
    return res_img

def contrast(gray, grad, inter):
    # res_img = gray.copy() * 0
    # your code here
    res_img = np.clip((grad * gray.copy()+inter), 0, 255).astype(np.uint8)
    return res_img

def scaling1(gray, s):
    h, w = gray.shape #(h:320, w:400)
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype) # 0으로 초기화된 matrics

    # your code here (uncomment below)
    #  forward warping
    # for r in range(h):
    #     for c in range(w):

    for r in range(h):
        for c in range(w):
            res_img[r*s][c*s] = gray[r][c]

    return res_img

def scaling2(gray, s):
    h, w = gray.shape
    res_img = np.zeros((int(h*s), int(w*s)), gray.dtype)

    # your code here (uncomment below)
    #  backward warping
    # for r in range(h*s):
    #     for c in range(w*s):
    for r in range(h*s):
        for c in range(w*s):
            res_img[r][c] = gray[math.floor(r/s)][math.floor((c/s))]

    return res_img

# deg: angle
def rotation(gray, radian):
    #res_img = gray.copy() * 0
    #  your code here
    h, w =gray.shape
    res_img = np.zeros((int(h), int(w)), gray.dtype)
    radian = radian * math.pi / 180

    rotation = np.array([[math.cos(radian), -1 * math.sin(radian)],
                       [math.sin(radian), math.cos(radian)]])

    for r in range(h):
        for c in range(w):
            data = rotation.dot(np.array([[r-h/2], [c-w/2]])) + np.array([[h/2],[w/2]])
            if 0 < data[0] < h and 0 < data[1] < w:
                res_img[r][c] = gray[math.floor(data[0])][math.floor(data[1])]

    return res_img




if __name__ == '__main__':

    # open image
    img_rgb = cv.imread('image.png', cv.IMREAD_COLOR)

    # get dimension shape 속성은 3개의 튜플을 전달한다
    # height width. y좌표 기준이라 h 부터 나온다
    # 채널은 그레이 채널이면 1이, rgb면 3채널
    h, w, ch = img_rgb.shape

    # if you want to know the dimension of img_rgb, remove comment below
    # print(img_rgb.shape) #-> (320, 400, 3)

    # mission 1 : convert color image to grayscale
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
    img_res1 = cv.hconcat([img_gray, img_bright, img_contrast])
    img_res2 = cv.hconcat([img_scaling1, img_scaling2])


    # display input image & results
    cv.imshow('input image', img_rgb)
    cv.imshow('gray, bright, contrast', img_res1)
    cv.imshow('scaling', img_res2)
    cv.imshow('rotation', img_rotation)
    cv.waitKey(0)