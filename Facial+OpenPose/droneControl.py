from djitellopy import Tello
import time
from threading import Thread
import keyboard
import cv2

keepRecording = True
recorder = 0

def initTello():
    myDrone = Tello()

    # drone connection
    if myDrone.connect():
        print("연결 문제")
        exit(0)

    # set all speed to 0
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0

    print("\n * Drone battery percentage: " + str(myDrone.get_battery()) + '%')
    # myDrone.streamoff()
    # time.sleep(1)

    return myDrone

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        time.sleep(1/30)

    video.release()

if __name__ == "__main__":
    myDrone = initTello()
    # myDrone.takeoff()
    # time.sleep(1)
    myDrone.streamon()
    cv2.namedWindow("drone")
    frame_read = myDrone.get_frame_read()
    time.sleep(2)

    while True:
        img = frame_read.frame
        cv2.imshow("Image", img)

        keyboard = cv2.waitKey(1)
        # Press "q" to exit
        if keyboard & 0xFF == ord('q'):
            # myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            keepRecording = False
            recorder.join()
            exit(0)
            break

        if keyboard == ord('v'):
            print('v')
            if recorder == 0:
                recorder = Thread(target=videoRecorder)
                recorder.start()

        if keyboard == ord('t'):
            cv2.imwrite("picture.png", frame_read.frame)
