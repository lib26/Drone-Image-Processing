import cv2
import numpy as np

# 전체적인 구성을 간단하게 설명하면 다음과 같다.
# Part1.py는 웹캠으로 사진을 100장 찍어 저장하고
# Part2.py는 저장한 사진을 학습해본다.
# 그리고 Part3.py에선 다시 저장한 사진을 학습 시킨 후 카메라 입력을 실시간으로 전달받아
# 동일 인물인지 아닌지 판단하여 UnLocked, Locked라고 화면에 표시해 주는 기능을 한다.


# 얼굴을 탐지할 수 있는 분류기를 만든다
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 원본 이미지에서 얼굴 부분만 빼온다
def face_extractor(img):

    # 하나의 frame 이미지를 회색처리한다
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 얼굴의 좌표 값, width, height 를 받는다.
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    # 속성값을 토대로 얼굴 부분 이미지를 생성한다.
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# 동영상 frame 을 받아온다. 웹캠인듯
cap = cv2.VideoCapture(0)
count = 0

while True:
    # 카메라로 부터 사진 1장 얻기
    ret, frame = cap.read()
    # 얼굴 감지 하여 얼굴만 가져오기
    if face_extractor(frame) is not None:
        count+=1
        # 가져온 얼굴 부분 이미지를 200 * 200 으로 resize
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # 얼굴 이미지 파일을 다른 faces path 에 저장한다.
        file_name_path = 'faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        # 화면에 얼굴과 현재 저장 개수 표시
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    # 100장 찍히면 종료
    if cv2.waitKey(1)==13 or count==100:
        break

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
