import numpy as np
import cv2

cap = cv2.VideoCapture(cv2.samples.findFile("video/scene2.avi"))
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    img2 = frame2.copy()
    step = 20
    for r in range(0,img2.shape[0], step):
        for c in range(0,img2.shape[1], step):
            offset_x = np.cos(ang[r,c]) * mag[r,c] * 2
            offset_y = np.sin(ang[r,c]) * mag[r,c] * 2
            cv2.line(img2, (c,r), (c-int(offset_x), r-int(offset_y)), (0, 255, 255), 2)

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('optical_flow', np.concatenate((img2, bgr), axis=1))

    if cv2.waitKey(30) == 27:
        break

    prvs = next