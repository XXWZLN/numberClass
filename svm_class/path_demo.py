import cv2
import os
from pathlib import Path

# 获取文件名
path_str = Path("./output/123.jpg")
path_file_name = path_str.name
print(path_file_name)

# 提取父文件路径
print(path_str.parent)

# 提取文件后缀
print(path_str.suffix)

# 提取无文件后缀名
print(path_str.stem)

# 遍历文件
path_str = Path("D:/OpenCV/DEEP/output")
path_all = os.listdir(path_str)
print(type(path_all))
# for path_all in os.listdir(path_str):
#     print(path_all)
#
# for path_all in path_str.iterdir():
#     print(path_all)

# 循环读取、保存图片测试
# 不使用os
# if 1 == 1:
#     for i in range(8):
#         readPath = "D:/OpenCV/DEEP/data/" + str(i + 1) + "/" + str(i + 1) + ".jpg"
#         writePath="D:/OpenCV/DEEP/output/"+str(i + 1) + ".jpg"
#         readName = str(i + 1) + ".jpg"
#         img = cv2.imread(readPath)
#         cv2.imshow(readName, img)
        #cv2.imwrite(writePath,img)

# 使用os(未完成，可能需要正则表达式)
if 1 == 0:
    readPath = "D:/OpenCV/DEEP/data"
    dirs = os.listdir(readPath)
    for file in dirs:
        pic_dir = os.path.join(readPath,file)



cv2.waitKey(0)
cv2.destroyAllWindows()
