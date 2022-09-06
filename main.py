import cv2 as cv
import sys

img = cv.imread("Lenna.png")
cv.imshow("Display Window", img)
cv.waitKey(0)