import cv2
from tensorflow import keras
from imutils.perspective import four_point_transform
import numpy as np

ans_count = 0
ans_last = 0
model = keras.models.load_model('my_model.h5')
cap = cv2.VideoCapture(1)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)
    cv2.imshow('canny',edged)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        peri = cv2.arcLength(cnts[0], True)
        approx = cv2.approxPolyDP(cnts[0], 0.02 * peri, True)
        pts = approx.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 225, 225))
        if len(approx) == 4:
            docCnt = approx
            docCnt = docCnt.reshape(4, 2)
            if docCnt is not None:
                paper = four_point_transform(frame, docCnt)
                h, w, c = paper.shape
                print("ps=",paper.shape)
                if w > 50 and h > 50:
                    gray_paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
                    paper_resized = cv2.resize(gray_paper,(32, 32))
                    input_image = paper_resized.reshape(1, 32, 32, 1)
                    input_image = input_image.astype('float32')
                    a = model.predict(input_image)
                    a = a.reshape(-1)
                    print(np.argmax(a) + 1)
    cv2.imshow('result', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
