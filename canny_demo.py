import cv2
from tensorflow import keras
from imutils.perspective import four_point_transform
import numpy as np

lowThreshold = 0


def nothing(x):
    pass


cap = cv2.VideoCapture(1)
cv2.namedWindow('canny demo')
cv2.createTrackbar('Min threshold', 'canny demo', 0, 200, nothing)
cv2.createTrackbar('Max threshold', 'canny demo', 0, 600, nothing)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Min_threshold=cv2.getTrackbarPos('Min threshold', 'canny demo')
    Max_threshold=cv2.getTrackbarPos('Max threshold', 'canny demo')
    img=cv2.Canny(gray,Min_threshold,Max_threshold)
    cv2.imshow('canny demo',img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
