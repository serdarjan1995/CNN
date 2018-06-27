import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

# Load an color image in grayscale
pathTestParking = 'test/parketme-durma/*.png'
pathTestWarning = 'test/tehlike-uyari/*.png'
pathLearningParking = 'egitim/parketme-durma/*.png'
pathLearningWarning = 'egitim/tehlike-uyari/*.png'

#img = cv2.imread('egitim/tehlike-uyari/2.png',0)
#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()
#plt.hist(img.ravel(),256,[0,256]); plt.show()

imagesLearningParking = []
img = cv2.imread("egitim/tehlike-uyari/13.png")
imgtest = cv2.imread("egitim/tehlike-uyari/2.png")
imgtest1 = cv2.imread("egitim/parketme-durma/3.png")
img2 = cv2.imread("egitim/tehlike-uyari/1.png")
img3 = cv2.imread("egitim/parketme-durma/1.png")

hist11 = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
hist22 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
hist33 = cv2.calcHist([img3], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])

histt11 = cv2.calcHist([imgtest], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
histt22 = cv2.calcHist([imgtest1], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])

d = cv2.compareHist(hist11, hist22, cv2.HISTCMP_CHISQR)
d1 = cv2.compareHist(hist11, hist11, cv2.HISTCMP_CHISQR)
d2 = cv2.compareHist(hist11, hist33, cv2.HISTCMP_CHISQR)
d3 = cv2.compareHist(hist33, hist22, cv2.HISTCMP_CHISQR)
dt1 = cv2.compareHist(hist11, histt11, cv2.HISTCMP_CHISQR)
dt2 = cv2.compareHist(hist33, histt22, cv2.HISTCMP_CHISQR)
dt3 = cv2.compareHist(hist11, histt22, cv2.HISTCMP_CHISQR)

hist0 = cv2.calcHist([imgtest1],[0],None,[256],[0,256])
hist1 = cv2.calcHist([imgtest1],[1],None,[256],[0,256])
hist2 = cv2.calcHist([imgtest1],[2],None,[256],[0,256])




BLUE_MIN = np.array([200, 100, 0], np.uint8)
BLUE_MAX = np.array([255, 255, 1], np.uint8)


dst = cv2.inRange(img3, BLUE_MIN, BLUE_MAX)
output = cv2.bitwise_and(img2, img3, dst = dst)
cv2.imshow("images", np.hstack([img3, output]))
cv2.waitKey(0)


cv2.imshow("images",dst)
cv2.waitKey(0)

for img in glob.glob(pathLearningParking):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    hist = cv2.calcHist([image],'b',None,[256],[0,256])
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()


for img in glob.glob(pathLearningWarning):
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    histr = cv2.calcHist([image],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.xlim([0,256])
plt.show()
    
    
    #imagesLearningParking.append(image)
print(img," read")
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
    #plt.hist(image.ravel(),256,[0,256]); plt.show()
    
imagesLearningWarning = []
for img in glob.glob(pathLearningWarning):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    imagesLearningWarning.append(image)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        plt.show()

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()