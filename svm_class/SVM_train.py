import numpy as np
import cv2
from sklearn import svm
import joblib
import os
from pathlib import Path
import time

filePath = './data'


# read all img path
def allFilePath(filePath):
    directory = Path(filePath)
    file_list = []
    for i in directory.iterdir():
        for j in i.iterdir():
            file_list.append(j)
    return file_list


def label_mat_create(file_list):
    label = []
    fileNumber = len(file_list)
    dataMat = np.zeros((fileNumber, 32 * 32))
    for i in range(fileNumber):
        # label
        fileName = file_list[i].stem
        label_num = fileName[0]
        label.append(label_num)
        # mat
        str_dir = str(file_list[i])
        # print(type(str_dir),str_dir)
        img = cv2.imread(str_dir, cv2.IMREAD_GRAYSCALE)
        ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
        img = np.reshape(img, (1, -1))
        dataMat[i, :] = img
    return label, dataMat


def create_svm(dataMat, dataLabel, path, decision='ovr'):
    clf = svm.SVC(C=1.0, kernel='rbf', decision_function_shape=decision)
    rf = clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path)
    return clf


if __name__ == '__main__':
    st = time.clock()
    file_list = allFilePath(filePath)
    label, dataMat = label_mat_create(file_list)
    path_model='./svm.model'
    create_svm(dataMat, label,path_model, decision='ovr')
    et = time.clock()
    print("Training spent {:.4f}s.".format((et - st)))