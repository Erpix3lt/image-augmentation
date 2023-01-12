import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

traindatagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=20, width_shift_range=0.3, height_shift_range=0.3)

image = cv.imread('images/Lena.jpg')
images = image[np.newaxis, :, :, :]




for i in range(6):
    # generate batch of images
    process_images = traindatagen.flow(images).next()
    process_image = process_images[0, :, :, :]
    cv.imwrite('test_process_image.jpg', process_image)
    cv.imwrite('lena' + str(i) + '.jpeg', process_image)  