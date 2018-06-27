# histogrami 
# test true ise tehlike-uyari sinifi
# test false ise parketme-durma sinifi
#

import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

pathTestParking = 'trafikisaretleri/test/parketme-durma/*.png'
pathTestWarning = 'trafikisaretleri/test/tehlike-uyari/*.png'
pathLearningParking = 'trafikisaretleri/egitim/parketme-durma/*.png'
pathLearningWarning = 'trafikisaretleri/egitim/tehlike-uyari/*.png'

imgsample = cv2.imread("trafikisaretleri/egitim/tehlike-uyari/2.png")
histSampleOfWarning = cv2.calcHist([imgsample], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])

#X = np.zeros((1,2), dtype='float32')
threshold_for_hist_diff = 2000;
threshold_for_blue_pixel = 20;
BLUE_MIN = np.array([200, 100, 0], np.uint8)
BLUE_MAX = np.array([255, 255, 1], np.uint8)



trainData = np.empty([0, 3], dtype='int32')
trainLabel = np.empty([0, 1], dtype='int32')
for file in glob.glob(pathLearningWarning):
    img = cv2.imread(file)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
    firstDiff = cv2.compareHist(hist, histSampleOfWarning, cv2.HISTCMP_CHISQR)
    if(firstDiff>threshold_for_hist_diff):
        firstDiff=1
    else:
        firstDiff=0
    dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
    secondDiff = np.mean(dst)
    if(secondDiff>threshold_for_blue_pixel):
        secondDiff=1
    else:
        secondDiff=0
    print(file,firstDiff,secondDiff)
    trainData = np.vstack((trainData,[firstDiff,secondDiff,1]))
    trainLabel = np.vstack((trainLabel,[1])) # 1 means warning
    
for file in glob.glob(pathLearningParking):
    img = cv2.imread(file)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
    firstDiff = cv2.compareHist(hist, histSampleOfWarning, cv2.HISTCMP_CHISQR)
    if(firstDiff>threshold_for_hist_diff):
        firstDiff=1
    else:
        firstDiff=0
    dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
    secondDiff = np.mean(dst)
    if(secondDiff>threshold_for_blue_pixel):
        secondDiff=1
    else:
        secondDiff=0
    print(file,firstDiff,secondDiff)
    trainData = np.vstack((trainData,[firstDiff,secondDiff,1]))
    trainLabel = np.vstack((trainLabel,[0])) # 0 means parking

w = np.zeros((3,))
for epoch in range(0,50):
    for row, label in zip(trainData, trainLabel):
        target = row.dot(w)>=0
        error = label - target
        w += error * row
        #print(w[0])
        #print()

for file in glob.glob(pathTestWarning):
    img = cv2.imread(file)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
    firstDiff = cv2.compareHist(hist, histSampleOfWarning, cv2.HISTCMP_CHISQR)
    if(firstDiff>threshold_for_hist_diff):
        firstDiff=1
    else:
        firstDiff=0
    dst = cv2.inRange(img, BLUE_MIN, BLUE_MAX)
    secondDiff = np.mean(dst)
    if(secondDiff>threshold_for_blue_pixel):
        secondDiff=1
    else:
        secondDiff=0
    #print(file,firstDiff,secondDiff)
    test = np.array([firstDiff,secondDiff,1],dtype='int32')
    target =test.dot(w)>=0
    print("test ",target)
