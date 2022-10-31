import numpy as np
import cv2

# feature xml 파일을 통해서 분류기를 만들고
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 이미지 회색 처리
img = cv2.imread('Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 얼굴 위치를 찾는다. 실제 결과 값을 받는다.
# 리턴값은 시작점과 끝점의 좌표, width, height
faces = face_cascade.detectMultiScale(gray)

#
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y,w,h), (255,0,255),2)
    # 원본 이미지에 각 속성들을 네모로 표시한다.
    roi_face = img[y:y+h, x:x+w]
    # roi : region of interest 관심있는 영역만 보겠다.
    # 원본 이미지에서 얼굴 내에서 눈을 찾기 위해 얼굴만 나오는 이미지를 뽑아낸다.
    # openCV에서는 y, x 좌표 순서임.
    cv2.imshow('roi', roi_face)
    roi_face = gray[y:y+h, x:x+w]
    # 원본에서 뽑아진 얼굴 이미지를 회색 처리한다.
    eyes = eye_cascade.detectMultiScale(roi_face)
    # roi 영역만 다시 한번 눈 속성을 추출한다.
    for (eye_x, eye_y, eye_w, eye_h) in faces:
        cv2.rectangle(img, (x+eye_x, y+eye_y, eye_w, eye_h), (0,255,0),2)
        # 원본 이미지에 눈을 표시하기 위해서는
        # x + eye_x 를 더해줘야함. 상대적인 거리이기 때문.

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()