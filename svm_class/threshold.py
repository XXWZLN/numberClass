import cv2
path='D:/OpenCV/DEEP/output/1/1_10.png'
img = cv2.imread(path)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("before", gray)
ret,gray=cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow("after", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()