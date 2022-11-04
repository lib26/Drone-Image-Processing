import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from droneControl import *
from djitellopy import Tello
import time
from threading import Thread

##얼굴
from os import listdir
from os.path import isfile, join


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



# 상승 : 오른팔 위로 일직선으로 들기
def check_up(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    noseIndex = [landmarks[0].x, landmarks[0].y]

    print(abs(right_elbow_Index[0] - right_Wrist_Index[0]))

    if (right_Wrist_Index[1] < noseIndex[1] and abs(right_elbow_Index[0] - right_Wrist_Index[0]) < 0.03): return True

    return False


# 착륙 : X
def check_down(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]

    if (right_Wrist_Index[0] > left_Wrist_Index[0]): return True

    return False


# 플립 : 동그라미
def check_o(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    left_elbow_Index = [landmarks[13].x, landmarks[13].y]

    noseIndex = [landmarks[0].x, landmarks[0].y]

    if (right_Wrist_Index[1] < noseIndex[1] and left_Wrist_Index[1] < noseIndex[1]
            and left_elbow_Index[0] - right_elbow_Index[0] > left_Wrist_Index[0] - right_Wrist_Index[0]): return True

    return False



myDrone=initTello()
time.sleep(5)
myDrone.streamon()
frame_read = myDrone.get_frame_read()
time.sleep(2)

# 이미지 파일의 경우 이것을 사용하세요.:
w = 640
h = 480
# 웹캠, 영상 파일의 경우 이것을 사용하세요.:




with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while frame_read:

        image = frame_read.frame

        #######얼굴 인식 시작
        # 얼굴 검출 시도
        image, face = face_detector(image)

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
            if confidence > 65:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                ############모션인식
                # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # 포즈 주석을 이미지 위에 그립니다.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('Face Cropper', image)
                # 보기 편하게 이미지를 좌우 반전합니다. -> 이거 근데 좀 렉걸림
                # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                ##############포즈 인식####################################################

                landmarks = results.pose_landmarks.landmark
                # print(landmarks)
                #
                # rightWristIndex = [landmarks[15].x, landmarks[15].y]
                # leftWristIndex = [landmarks[16].x, landmarks[16].y]
                # print(rightWristIndex)
                #
                # rightWrist = [rightWristIndex[0] * w, rightWristIndex[1] * h]
                # leftWrist = [leftWristIndex[0] * w, leftWristIndex[1] * h]
                if check_up(landmarks):
                    print('드론 상승')
                    time.sleep(3)
                elif check_down(landmarks):
                    print('착륙')
                    time.sleep(3)
                elif check_o(landmarks):
                    print('플립')
                    time.sleep(3)



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


# cap.release()