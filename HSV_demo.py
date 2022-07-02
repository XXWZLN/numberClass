import cv2
import numpy as np

cap = cv2.VideoCapture(1)


def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 0, 180, nothing)
cv2.createTrackbar("US", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 0, 255, nothing)

while True:
    # 读取图片
    _, frame = cap.read()
    # 压缩图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_g = np.array([l_h, l_s, l_v])  # lower green value
    u_g = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_g, u_g)

    res = cv2.bitwise_and(frame, frame, mask=mask)  # src1,src2

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(1)
    if key == 27:  # Esc
        break

cv2.destroyAllWindows()
