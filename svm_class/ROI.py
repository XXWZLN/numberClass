import cv2
import numpy as np
from tensorflow import keras
from imutils.perspective import four_point_transform

# HSV域
l_g = np.array([0, 0, 0])
u_g = np.array([180, 255, 115])
peri = 10
j = 4


def nothing():
    pass


# HSV域调整时开启
if 1 == 0:
    # HSV滑块定义
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 0, 180, nothing)
    cv2.createTrackbar("US", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 0, 255, nothing)


# HSV调整函数
def HSV_config():
    # 滑块数值读取
    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")
    # 滑块数值修改相应HSV值
    l_g_c = np.array([l_h, l_s, l_v])
    u_g_c = np.array([u_h, u_s, u_v])
    return l_g_c, u_g_c


# 图像处理函数
def imageProcessing(img):
    # 膨胀算法，设置4*4方形卷积核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilation = cv2.dilate(img, kernel)
    cv2.imshow('dilation', img)
    return img


# 数字ROI分割，原始训练图像保存函数
def ROIdef_and_save(img):
    # 轮廓识别
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(frame, contours, -1, (0, 0, 255))
    if len(contours) > 0:
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > 19000:
                # 将传入轮廓多边形拟合，返回多边形点
                approx = cv2.approxPolyDP(contours[i], peri, True)
                # 连点成线，绘制多边形
                cv2.polylines(approxs, [approx], True, (255, 0, 0), 1)
                if approx.shape[0] == 4:
                    approx = approx.reshape(4, 2)
                    if approx is not None:
                        transformed_gray_single_num = four_point_transform(img, approx)
                        cv2.imshow("1", transformed_gray_single_num)


def removeFrame(img):
    for i in range(32):
        if i == 0 or i == 31:
            for j in range(32):
                img[i][j] = 0
        else:
            img[i][0] = 0
            img[i][31] = 0
    return img


frame = cv2.imread('567.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# HSV域调整时开启
if 1 == 0:
    l_g, u_g = HSV_config()
mask = cv2.inRange(gray, l_g, u_g)
cv2.imshow("mask", mask)
# 图像优化处理开启
if 1 == 0:
    imageProcessing(mask)
# ROIdef_and_save(mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 绘制轮廓
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
                if approx is not None:
                    writePath = "D:/OpenCV/DEEP/output/" + str(j + 1) + ".jpg"
                    transformed_gray_single_num = four_point_transform(mask, approx)
                    # cv2.imshow(str(i), transformed_gray_single_num)
                    transformed_gray_single_num = cv2.resize(transformed_gray_single_num, (32, 32))
                    train_img=removeFrame(transformed_gray_single_num)
                    cv2.imshow(str(i),train_img)
                    # cv2.imwrite(writePath, transformed_gray_single_num)
                    j = j + 1

cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

