from utils import *
import time
import cv2
from threading import Thread
import keyboard

keepRecording = True
recorder = 0

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
    myDrone.takeoff()
    time.sleep(1)
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
            myDrone.land()
            frame_read.stop()
            myDrone.streamoff()
            keepRecording = False
            recorder.join()
            exit(0)
            break

        if keyboard.is_pressed('w'):
            time.sleep(1)
            myDrone.move_forward(20)
        if keyboard.is_pressed('s'):
            time.sleep(1)
            myDrone.move_back(20)
        if keyboard.is_pressed('a'):
            time.sleep(1)
            myDrone.move_left(20)
        if keyboard.is_pressed('d'):
            time.sleep(1)
            myDrone.move_right(70)
        if keyboard == ord('v'):
            if recorder == 0:
                recorder = Thread(target=videoRecorder)
                recorder.start()

