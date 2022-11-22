import cv2
from pathlib import Path
import threading
import numpy as np
from os import listdir
from os.path import isfile, join

from droneControl import *
from djitellopy import Tello
import time
from threading import Thread
import keyboard

##### 미디어파이프
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 이미지 파일의 경우 이것을 사용하세요.:
w = 640
h = 480

# 전반적인 프로그램 실행 순서 ( 미완 )

# 전체 body를 detect
# 각 body에서 face를 detect
# detect한 face에서 predict(recognition)
# predict 성공한 face와 body 정보를 알아내고
# 그 body 만큼만 frame을 잘라서 openpose

# 팔 위로 올리기 - 고도 상승
# 팔 앞으로 뻗기 - 전진
# 팔 옆으로 굽혀서 위로 올리기 - 후진
# X 표시 - 착륙

# 손흥민 - 파노라마, 찰칵 -> 사진 전송
# 동그라미 - 플립


# 여기부터 part2와 동일
data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")

# 여긴 part1과 거의 동일
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# part1에서는 extractor 였지만 여기선 얼굴 감별
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi
# 여기까지 part1과 거의 동일



time.sleep(5)
myDrone = initTello()
frame_read = myDrone.get_frame_read()
time.sleep(2)

# 카메라 열기

while True:
    # 카메라로 부터 사진 한장 읽기
    frame = frame_read.frame

    # 얼굴 검출 시도
    image, face = face_detector(frame)

    try:
        # 검출된 사진 흑백으로 변환
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 위에서 학습한 모델로 예측시도
        result = model.predict(face)

        # result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다.
        if result[1] < 500:
            # ???? 어쨋든 0-100사이로 표시하려고 한듯
            confidence = int(100*(1-(result[1])/300))

            # 유사도 화면에 표시
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

        # 75 보다 크면 동일 인물로 간주해 UnLocked!
        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)

            ################################################

            with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as pose:

                    # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    rightindex = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
                    leftindex = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

                    righthand = [rightindex[0], rightindex[1]]
                    lefthand = [leftindex[0] * w, leftindex[1] * h]

                    print(righthand)
                    # landmarks = results.pose_landmarks.landmark
                    #
                    # rightwrist = [landmarks[16].x, landmarks[16].y]
                    # leftwrist = [landmarks[15].x, landmarks[15].y]
                    # rightwrist = [rightwrist[0] * w, rightwrist[1] * h]
                    # leftwrist = [leftwrist[0] * w, leftwrist[1] * h]
                    #
                    #
                    # print(leftwrist[0])
                    # # if leftwrist[0] > rightwrist[0]:
                    # #     print('nice')

                    # 포즈 주석을 이미지 위에 그립니다.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # 보기 편하게 이미지를 좌우 반전합니다.

                    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

            ################################################

        # 75 이하면 타인.. Locked!
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        # 얼굴 검출 안됨
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


frame_read.stop()
myDrone.streamoff()
cv2.destroyAllWindows()
exit(0)
