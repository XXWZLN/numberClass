import cv2
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'D:/OpenCV/DEEP/output'
myList = os.listdir(path)
Classes = len(myList)
gen = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.1,
                         rotation_range=10
                         )


def image_create(img_dir, save_dir, num=20):
    x = cv2.imread(img_dir, 0)
    x = x.reshape((1,) + x.shape + (1,))
    img_flow = gen.flow(
        x,
        batch_size=1,
        save_to_dir=save_dir,
        save_prefix="data",
        save_format="png"
    )
    i = 0
    for batch in img_flow:
        i += 1
        if i > num:
            break


# for i in range(2,8):
#     image_create(path + "/" + str(i)+"/"+str(i)+".jpg", path + "/"+str(i))
i=8
image_create(path + "/" + str(i)+"/"+str(i)+".jpg", path + "/"+str(i))
