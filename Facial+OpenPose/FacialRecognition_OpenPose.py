import cv2
from pathlib import Path
import threading
import numpy as np
from os import listdir
from os.path import isfile, join

# Face Recognition Setting Variables
data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi


# OpenPose Setting Variables
# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# 각 파일 path
BASE_DIR = Path(__file__).resolve().parent
protoFile = str(BASE_DIR) + "/poses/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = str(BASE_DIR) + "/poses/pose_iter_160000.caffemodel"
pose_type = "XHandsUp"
# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
status = 2
keep_time = [0, 0, 0]
O_X_HandsUp = ["O", "X", "HandsUp"]
Success_Fail = ["Success", "Fail"]

return_result = ""


############################################
# 1초마다 자세를 체크하도록 코드를 짰다. => threading.Timer 사용

def check_timer1(pose_type):  # O,X,HandsUp
    global keep_time, status
    global return_result
    if keep_time[0] == 5 or keep_time[1] == 5 or keep_time[2] == 8:
        result = show_result(pose_type, status)  # O,X,HandsUp / Again
        if result in O_X_HandsUp:
            return_result = result
        elif result == "Again":
            keep_time[0] = keep_time[1] = keep_time[2] = 0  # 시간 초기화

    elif keep_time[status] == 0:
        # 다른 모션에서 바뀌어 들어옴
        keep_time[0] = keep_time[1] = keep_time[2] = 0
        keep_time[status] += 1

    elif keep_time[status] != 0:
        keep_time[status] += 1

    print(keep_time[0], keep_time[1], keep_time[2])

    if return_result == "":
        threading.Timer(1.0, check_timer1, (pose_type,)).start()


def check_timer2(pose_type):  # Stretching
    global keep_time, status
    global return_result
    if keep_time[0] == 10 or keep_time[2] == 8:  # 자세유지 10초 하면 끝
        result = show_result(pose_type, status)  # Success / Fail
        return_result = result

    elif keep_time[status] == 0:
        # 다른 모션에서 바뀌어 들어옴
        keep_time[0] = keep_time[2] = 0
        keep_time[status] += 1

    elif keep_time[status] != 0:
        keep_time[status] += 1

    print(keep_time[0], keep_time[1], keep_time[2])

    if return_result == "":
        threading.Timer(1.0, check_timer2, (pose_type,)).start()


##################################################
# 자세 체크 함수들

def check_HandsUp(points):
    if points[0] and points[2] and points[3] and points[5] and points[6]:
        head_x, head_y = points[0]  # 머리
        rs_x, rs_y = points[2]  # 오른쪽 어깨
        re_x, re_y = points[3]  # 오른쪽 팔꿈치
        ls_x, ls_y = points[5]  # 왼쪽 어깨
        le_x, le_y = points[6]  # 왼쪽 팔꿈치

        # 팔꿈치가 어깨보다 높을 것, 양 팔꿈치 사이에 머리가 위치할 것
        if re_y < rs_y and le_y < ls_y and re_x < head_x and head_x < le_x:

            if points[4] and points[7]:
                rw_x, rw_y = points[4]  # 오른쪽 손목
                lw_x, lw_y = points[7]  # 왼쪽 손목

                # 양쪽 손목 중, 어느 하나라도 머리보다는 위에 가야한다.
                if rw_y < head_y or lw_y < head_y:
                    return True
                else:
                    return False

            else:
                return False

        else:
            return False

    else:
        return False


def check_O(points):
    if points[3] and points[4] and points[6] and points[7]:
        re_x, re_y = points[3]  # 오른쪽 팔꿈치
        rw_x, rw_y = points[4]  # 오른쪽 손목
        le_x, le_y = points[6]  # 왼쪽 팔꿈치
        lw_x, lw_y = points[7]  # 왼쪽 손목

        # 기본적으로 만세 조건을 만족시킬것 check_HandsUP()
        # 손목이 팔꿈치보다 안쪽에 위치할 것
        # 손목이 팔꿈치보다 위쪽에 위치할 것
        if check_HandsUp(points) and ((re_x < rw_x and rw_y < re_y) or (le_x > lw_x and le_y > lw_y)):
            return True
        else:
            return False

    else:
        return False


def check_X(points):
    if points[0] and points[2] and points[3] and points[4] and points[5] and points[6] and points[7] and points[8]:
        head_x, head_y = points[0]  # 머리
        rs_x, rs_y = points[2]  # 오른쪽 어깨
        re_x, re_y = points[3]  # 오른쪽 팔꿈치
        rw_x, rw_y = points[4]  # 오른쪽 손목
        ls_x, ls_y = points[5]  # 왼쪽 어깨
        le_x, le_y = points[6]  # 왼쪽 팔꿈치
        lw_x, lw_y = points[7]  # 왼쪽 손목
        b_x, b_y = points[8]  # 골반(오른쪽)

        # 골반보다 팔꿈치가 위쪽에 위치 and 팔꿈치보다 손목이 위쪽에 위치 and 손목보다 머리가 위쪽에 위치
        # 어깨 안쪽으로 손목이 위치
        if (b_y > le_y and b_y > re_y) and (le_y > lw_y and re_y > rw_y) and (lw_y > head_y and rw_y > head_y):
            if rs_x < rw_x or lw_x < ls_x:
                r_gradient = -1
                l_gradient = 1
                if rw_x - re_x != 0:
                    r_gradient = (rw_y - re_y) / (rw_x - re_x)
                if lw_x - le_x != 0:
                    l_gradient = (lw_y - le_y) / (lw_x - le_x)

                if r_gradient < 0 or l_gradient > 0:
                    return True
                else:
                    return False
    else:
        return False


def check_Stretching(points):
    # 오른쪽 골반, 오른쪽 무릎, 오른쪽 발목,  왼쪽 골반, 왼쪽 무릎, 왼쪽 발목
    if points[8] and points[9] and points[10] and points[11] and points[12] and points[13]:
        rh_x, rh_y = points[8]  # 오른쪽 골반
        rk_x, rk_y = points[9]  # 오른쪽 무릎
        ra_x, ra_y = points[10]  # 오른쪽 발목
        lh_x, lh_y = points[11]  # 왼쪽 골반
        lk_x, lk_y = points[12]  # 왼쪽 무릎
        la_x, la_y = points[13]  # 왼쪽 발목

        # 기본적으로 O 조건을 만족시킬 것 check_O()
        if check_O(points):
            # 오른쪽을 구부림
            if rk_x < rh_x and rk_x < ra_x and ra_y < la_y:
                return True
            # 왼쪽을 구부림
            elif lk_x > lh_x and lk_x > la_x and la_y < ra_y:
                return True
        else:
            return False

    else:
        return False


#################################################
# 결과를 반환하는 함수

def show_result(pose_type, status):  # END/Again
    if status == 2:  # status 2
        if pose_type == "Stretching":
            print("자세 유지에 실패하셨습니다.")
            return "Fail"
        else:
            print("자세를 취해주세요.")
            return "Again"

    elif status == 1:  # status 1
        print("X를 선택하셨습니다.")
        return "X"

    else:  # status 0
        if pose_type == "OX":
            print("O를 선택하셨습니다.")
            return "O"

        elif pose_type == "XHandsUp":
            print("만세를 선택하셨습니다.")
            return "HandsUp"

        elif pose_type == "Stretching":
            print("자세유지에 성공하셨습니다.")
            return "Success"


##################################################
# 쿠다 사용 안하면 밑에 이미지 크기를 줄이는게 나을 것이다
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #벡엔드로 쿠다를 사용하여 속도향상을 꾀한다
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # 쿠다 디바이스에 계산 요청


###카메라랑 연결
capture = cv2.VideoCapture(0)  # 카메라 정보 받아옴
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #카메라 속성 설정
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # width:너비, height: 높이

inputWidth = 320;
inputHeight = 240;
inputScale = 1.0 / 255;

# 반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while cv2.waitKey(1) < 0:  # 아무 키나 누르면 끝난다.

    # Face Recognition
    ret, frame = capture.read()
    # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not ret:
        cv2.waitKey()
        break

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + '% Confidence it is user'
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

        if confidence > 80:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            #
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False,
                                            crop=False)

            imgb = cv2.dnn.imagesFromBlob(inpBlob)
            # cv2.imshow("motion",(imgb[0]*255.0).astype(np.uint8))

            # network에 넣어주기
            net.setInput(inpBlob)

            # 결과 받아오기
            output = net.forward()

            # 키포인트 검출시 이미지에 그려줌
            points = []
            for i in range(0, 15):
                # 해당 신체부위 신뢰도 얻음.
                probMap = output[0, i, :, :]

                # global 최대값 찾기
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                # 원래 이미지에 맞게 점 위치 변경
                x = (frameWidth * point[0]) / output.shape[3]
                y = (frameHeight * point[1]) / output.shape[2]

                # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
                if prob > 0.1:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
                    cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                lineType=cv2.LINE_AA)
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
            for pair in POSE_PAIRS:
                partA = pair[0]  # Head
                partA = BODY_PARTS[partA]  # 0
                partB = pair[1]  # Neck
                partB = BODY_PARTS[partB]  # 1

                # partA와 partB 사이에 선을 그어줌 (cv2.line)
                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
                    # OX일때
                    if pose_type == "OX":
                        if check_O(points):
                            status = 0
                            cv2.putText(frame, "choose O", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                        elif check_X(points):
                            status = 1
                            cv2.putText(frame, "choose X", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                        else:
                            status = 2
                            cv2.putText(frame, "None", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)

                    # XHandsUp일때
                    elif pose_type == "XHandsUp":
                        if check_HandsUp(points):
                            cv2.putText(frame, "choose HandsUp", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                            status = 0
                        elif check_X(points):
                            status = 1
                            cv2.putText(frame, "choose X", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                        else:
                            status = 2
                            cv2.putText(frame, "None", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)

                    # Stretching일때
                    elif pose_type == "Stretching":
                        if check_Stretching(points):
                            status = 0
                            cv2.putText(frame, "Keep going", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)
                        else:
                            status = 2
                            cv2.putText(frame, "None", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 1,
                                        lineType=cv2.LINE_AA)

                    # 결과가 결정이 나면 반환
                    if return_result in O_X_HandsUp or return_result in Success_Fail:
                        print('RETURN RESULT:',return_result)

            cv2.imshow("Output-Keypoints", frame)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1) == 13:
        break


capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 윈도우 창 닫음