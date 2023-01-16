import numpy as np
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
import os

batchSize = 10
fileName = "Lena.jpg"
inputDirectory = "images/input/"
outputDirectory = "images/output/"

#--- ImageDataGenerator Args---#
featureWiseCenter = False
featureWiseSTDNormalization = False
rotationRange = 20
widthShiftRange = 0.3
heightShiftRange = 0.3
brightnessRange = (0.7, 1)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

traindatagen = ImageDataGenerator(featurewise_center=featureWiseCenter, featurewise_std_normalization=featureWiseSTDNormalization,
                                  rotation_range=rotationRange, width_shift_range=widthShiftRange, height_shift_range=heightShiftRange, brightness_range=brightnessRange)

allInputImages = load_images_from_folder(inputDirectory)

for index, image in enumerate(allInputImages):
    images = image[np.newaxis, :, :, :]

    for i in range(batchSize):
        # generate batch of images
        processimages = traindatagen.flow(images).next()
        processimage = processimages[0, :, :, :]
        cv.imwrite(outputDirectory + str(index) + "_version" + str(i) + ".jpg", processimage)
    
    cv.imwrite(outputDirectory + str(index) + "_original_version" + ".jpg", image)

