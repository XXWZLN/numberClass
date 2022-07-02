import cv2

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()
    print(frame.shape)
    # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
    cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.imwrite("123.jpg", frame)
        print("done")
        break

cv2.destroyAllWindows()
