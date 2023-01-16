import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

traindatagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, 
                                rotation_range=20, width_shift_range=0.3, height_shift_range=0.3, brightness_range=0.4)

image = cv.imread('images/lena.jpg')
images = image[np.newaxis, :, :, :]

for i in range(6):
    # generate batch of images
    process_images = traindatagen.flow(images).next()
    process_image = process_images[0, :, :, :]
    cv.imwrite('images/output/lena' + str(i) + '.jpg', process_image)  