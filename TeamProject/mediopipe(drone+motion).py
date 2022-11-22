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


# myDrone=initTello()
# time.sleep(5)
# myDrone.streamon()
# frame_read = myDrone.get_frame_read()
# time.sleep(2)

# 상승 : O
def check_up(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    left_elbow_Index = [landmarks[13].x, landmarks[13].y]

    noseIndex = [landmarks[0].x, landmarks[0].y]

    if (right_Wrist_Index[1] < noseIndex[1] and left_Wrist_Index[1] < noseIndex[1]
            and left_elbow_Index[0] - right_elbow_Index[0] > left_Wrist_Index[0] - right_Wrist_Index[0]): return True

    return False



# 착륙 : X
def check_down(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]

    if (right_Wrist_Index[0] > left_Wrist_Index[0]): return True

    return False

# 전진 : 오른팔 뻗기
def check_go(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    noseIndex = [landmarks[0].x, landmarks[0].y]

    if (right_Wrist_Index[1] < noseIndex[1] and abs(right_elbow_Index[0] - right_Wrist_Index[0]) < 0.03): return True

    return False


# 후진 : 왼팔 뻗기
def check_back(landmarks):
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]
    left_elbow_Index = [landmarks[13].x, landmarks[13].y]
    noseIndex = [landmarks[0].x, landmarks[0].y]

    if (left_Wrist_Index[1] < noseIndex[1] and abs(left_elbow_Index[0] - left_Wrist_Index[0]) < 0.03): return True

    return False


# 플립 : Y
def check_flip(landmarks):
    right_Wrist_Index = [landmarks[16].x, landmarks[16].y]
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    left_Wrist_Index = [landmarks[15].x, landmarks[15].y]
    left_elbow_Index = [landmarks[13].x, landmarks[13].y]
    noseIndex = [landmarks[0].x, landmarks[0].y]

    if (left_Wrist_Index[1] < noseIndex[1] and right_Wrist_Index[1] < noseIndex[1] and
            abs(left_Wrist_Index[0] - right_Wrist_Index[0]) > abs(
                left_elbow_Index[0] - right_elbow_Index[0])): return True

    return False

# 사진 : 손흥민
def check_take_picture(landmarks):
    right_elbow_Index = [landmarks[14].x, landmarks[14].y]
    left_elbow_Index = [landmarks[13].x, landmarks[13].y]
    right_shoulder = [landmarks[11].x, landmarks[11].y]

    print(abs(right_elbow_Index[0] - left_elbow_Index[0]))
    #
    # if (right_elbow_Index[1] > right_shoulder[1] and left_elbow_Index[1] > right_shoulder[1]
    #     and abs(right_elbow_Index[0]-left_elbow_Index[0]) ): return True

    return False



# 웹캠, 영상 파일의 경우 이것을 사용하세요.:
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue

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

        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

        ##############포즈 인식####################################################

        # landmarks = results.pose_landmarks.landmark
        # print(landmarks)
        #
        # if check_up(landmarks):
        #     print('상승')
        #     time.sleep(3)
        # elif check_down(landmarks):
        #     print('착륙')
        #     time.sleep(3)
        # elif check_go(landmarks):
        #     print('전진')
        #     time.sleep(3)
        # elif check_back(landmarks):
        #     print('후진')
        #     time.sleep(3)
        # elif check_flip(landmarks):
        #     print('플립')
        #     time.sleep(3)
        # elif check_take_picture(landmarks):
        #     print('사진')
        #     time.sleep(3)

cap.release()
