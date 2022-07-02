import cv2
import os

from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def callback1(s):
    return int(s) - 1


def DataImport(path='data', valRatio=0.2, testRatio=0.1):
    images = []
    imagesClass = []
    Classes = os.listdir(path)
    print(Classes)
    for x in Classes:
        dataPath = os.listdir(path + "/" + x)
        for y in dataPath:
            img = cv2.imread(path + "/" + x + "/" + y, 0)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            imagesClass.append(x)
    images = np.array(images)
    x_train, x_test, y_train, y_test = train_test_split(images, imagesClass, test_size=valRatio)
    y_train = list(map(callback1, y_train))
    y_test = list(map(callback1, y_test))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=valRatio)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_validation = x_validation.reshape(x_validation.shape + (1,))
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)
    return x_train, x_test, x_validation, y_train, y_test, y_validation


def MYmodel():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(120, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


x_train, x_test, x_validation, y_train, y_test, y_validation = DataImport()
model = MYmodel()
print(model.summary())

trainpicGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                                 rotation_range=10)
model.fit_generator(trainpicGen.flow(x_train, y_train, batch_size=16),
                    steps_per_epoch=625,
                    epochs=10,
                    validation_data=(x_validation, y_validation),
                    validation_steps=100,
                    shuffle=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy =', score[1])
model.save('my_model.h5')
