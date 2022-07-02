import cv2
import numpy as np

img = cv2.imread("D:/OpenCV/DEEP/output/1.jpg")
for i in range(32):
    if i==0 or i==31:
        for j in range(32):
            img[i][j]=255
    else:
        img[i][0]=255
        img[i][31]=255
cv2.imshow("img",img)

cv2.waitKey(0)
cv2.destroyAllWindows()