import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import os

batchSize = 10
fileName = "Lena.jpg"
inputDirectory = "images"
outputDirectory = "images/output/"

#--- ImageDataGenerator Args---#
featureWiseCenter = False
featureWiseSTDNormalization = False
rotationRange = 20
widthShiftRange = 0.3
heightShiftRange = 0.3
brightnessRange = (0.7, 1)



traindatagen = ImageDataGenerator(featurewise_center=featureWiseCenter, featurewise_std_normalization=featureWiseSTDNormalization,
                                  rotation_range=rotationRange, width_shift_range=widthShiftRange, height_shift_range=heightShiftRange, brightness_range=brightnessRange)

image = cv.imread(inputDirectory + "/" + fileName)
images = image[np.newaxis, :, :, :]

for i in range(batchSize):
    # generate batch of images
    processimages = traindatagen.flow(images).next()
    processimage = processimages[0, :, :, :]
    cv.imwrite(outputDirectory + "/" + str(i) + fileName, processimage)
