import cv2
import numpy as np
from imutils.perspective import four_point_transform
import time
from sklearn import svm
import joblib

cap = cv2.VideoCapture(1)
l_g = np.array([0, 0, 0])  # lower green value
u_g = np.array([180, 255, 115])
font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
model_path='./svm.model'
clf = joblib.load(model_path)
peri = 10

def nothing(x):
    pass

def removeFrame(img):
    for i in range(32):
        if i == 0 or i == 31:
            for j in range(32):
                img[i][j] = 0
        else:
            img[i][0] = 0
            img[i][31] = 0
    return img


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    contour = frame.copy()
    mask = cv2.inRange(gray, l_g, u_g)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255))
    if len(contours) > 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 16000:
                # 将传入轮廓多边形拟合，返回多边形点
                approx = cv2.approxPolyDP(contours[i], peri, True)
                # 连点成线，绘制多边形
                cv2.polylines(frame, [approx], True, (255, 0, 0), 1)
                if approx.shape[0] == 4:

                    approx = approx.reshape(4, 2)
                    cv2.circle(frame, approx[0], 1, (0, 0, 255),2)
                    if approx is not None:
                        transformed_gray_single_num = four_point_transform(mask, approx)
                        transformed_gray_single_num = cv2.resize(transformed_gray_single_num, (32, 32))
                        train_img=removeFrame(transformed_gray_single_num)
                        ret, img = cv2.threshold(train_img, 160, 255, cv2.THRESH_BINARY)
                        img = np.reshape(img, (1, -1))
                        preResult = clf.predict(img)
                        preResult=str(preResult)
                        print(preResult)
                        cv2.putText(frame, preResult[2], approx[0], font, 1.2, (0, 0, 0), 2)

    cv2.imshow('contours', contour)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
