import numpy as np
import cv2
from sklearn import svm
import joblib
import os
from pathlib import Path
import time

path_num = ['D:/OpenCV/DEEP/output/1/1_4.png',
            'D:/OpenCV/DEEP/output/2/2_5.png',
            'D:/OpenCV/DEEP/output/4/4_7.png',
            'D:/OpenCV/DEEP/output/4/4_1.png',
            'D:/OpenCV/DEEP/output/5/5_16.png',
            'D:/OpenCV/DEEP/output/5/5_17.png',
            'D:/OpenCV/DEEP/output/7/7_18.png',
            'D:/OpenCV/DEEP/output/8/8_10.png',
            ]  # 需要更改
model_path = './svm.model'
clf = joblib.load(model_path)
mat = np.zeros((8, 32 * 32))
for i in range(8):
    img = cv2.imread(path_num[i], cv2.IMREAD_GRAYSCALE)
    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (1, -1))
    mat[i] = img
preResult = clf.predict(mat)
print(preResult)
