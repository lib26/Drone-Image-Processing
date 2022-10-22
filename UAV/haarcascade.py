import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y,w,h), (255,0,255),2)
    roi_face = img[y:y+h, x:x+w]
    cv2.imshow('roi', roi_face)
    roi_face = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_face)
    for (eye_x, eye_y, eye_w, eye_h) in faces:
        cv2.rectangle(img, (x+eye_x, y+eye_y, eye_w, eye_h), (0,255,0),2)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()