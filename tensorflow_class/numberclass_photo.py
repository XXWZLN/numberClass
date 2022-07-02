import cv2
import numpy as np
from tensorflow import keras
from imutils.perspective import four_point_transform
import time

l_g = np.array([0, 0, 0])  # lower green value
u_g = np.array([180, 255, 115])
model = keras.models.load_model('my_model.h5')
font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体



def nothing():
    pass


peri = 10

while True:
    frame = cv2.imread('123.jpg')
    # frame = frame[0:480,160:480]
    contour = frame.copy()
    mask = cv2.inRange(frame, l_g, u_g)
    approxs = frame.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilation = cv2.dilate(mask, kernel)
    cv2.imshow('dilation', dilation)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour, contours, -1, (0, 0, 255))
    if len(contours) > 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 19000:
                cv2.drawContours(contour, contours, i, (0, 225, 0), 3)
                approx = cv2.approxPolyDP(contours[i], peri, True)
                print(approx.shape[0])
                cv2.polylines(approxs, [approx], True, (255, 0, 0), 1)
                if approx.shape[0] == 4:
                    docCnt = approx
                    docCnt = docCnt.reshape(4, 2)
                    cv2.circle(approxs, docCnt[0], 1, (0, 0, 255),2)
                    if docCnt is not None:
                        paper = four_point_transform(mask, docCnt)
                        paper_resized = cv2.resize(paper, (32, 32))
                        input_image = paper_resized.reshape(1, 32, 32, 1)
                        input_image = input_image.astype('float32')
                        a = model.predict(input_image)
                        a = a.reshape(-1)
                        print("number=",np.argmax(a) + 1)

                        approxs = cv2.putText(approxs, str(np.argmax(a) + 1), docCnt[0], font, 1.2, (0, 0, 0), 2)

    cv2.imshow('contours', contour)
    cv2.imshow('approx', approxs)

    cv2.waitKey(0)
