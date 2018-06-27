# resimler için kullanılan özellikler [4]:  histogram(RGB),
#                                           LBP (her class için 1 örnek)
#                                           SURF (keypoint sayısı),
#                                           resimlerin yeşil pixel seviyesi
#
#
#
#
#
#


import numpy as np
import cv2, glob
from skimage.feature import local_binary_pattern
from scipy.spatial import distance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.layers.advanced_activations import PReLU
import time

path_train = [
    'artpictures/train/action/',
    'artpictures/train/colorful/',
    'artpictures/train/landscape/',
    'artpictures/train/modern/'
    ]

path_test = [
    'artpictures/test/action/',
    'artpictures/test/colorful/',
    'artpictures/test/landscape/',
    'artpictures/test/modern/'
    ]

labels = [
    'action',
    'colorful',
    'landscape',
    'modern'
    ]

lbp_radius = 1
lbp_points = lbp_radius * 8
lbp_method = 'uniform'
surf_thresh = 2000

GREEN_MIN = np.array([0, 70, 20], np.uint8)
GREEN_MAX = np.array([50, 255, 110], np.uint8)

histogram_bins = [5, 5, 5]

histogram_avg = np.zeros([len(path_train), 5,5,5], dtype='float32')
lbp_histogram_samples = np.zeros([len(path_train),lbp_points+2], dtype='float64')
green_level_mean = np.zeros(len(path_train))


for i in range(len(path_train)):
    print('\n[INFO]Feature extraction => Loading files: ',path_train[i])
    fileCount = len(glob.glob1(path_train[i],'*.jpg'))
    print(path_train[i])
    for file in glob.glob1(path_train[i],'*.jpg'):
        #print('file: ',file)
        img = cv2.imread(path_train[i]+file)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        #first attribute: histogram average
        hist = cv2.calcHist([img], [0, 1, 2], None, histogram_bins,[0, 256, 0, 256, 0, 256])
        histogram_avg[i] += hist
    histogram_avg[i] /= fileCount

#third attribute: Local Binary Pattern (texture) histogram for 2 random image in dir
#action
img_gray = cv2.imread(path_train[0]+'6.jpg',0)
lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2, range=(0, lbp_points + 2))
lbp_histogram_samples[0] = lbp_hist

#colorful
img_gray = cv2.imread(path_train[1]+'2.jpg',0)
lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2, range=(0, lbp_points + 2))
lbp_histogram_samples[1] = lbp_hist

#landscape
img_gray = cv2.imread(path_train[2]+'3.jpg',0)
lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2, range=(0, lbp_points + 2))
lbp_histogram_samples[2] = lbp_hist

#modern
img_gray = cv2.imread(path_train[3]+'6.jpg',0)
lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2, range=(0, lbp_points + 2))
lbp_histogram_samples[3] = lbp_hist



print('\n[INFO]Feature extraction completed!\n')
## feature extraction completed

trainData = np.empty([0, 5], dtype='int32')
trainLabel = np.empty([0, 4], dtype='int32')

for i in range(len(path_train)):
    print('\n[INFO]Train Data creation => Loading files: ',path_train[i])
    fileCount = len(glob.glob1(path_train[i],'*.jpg'))
    for file in glob.glob1(path_train[i],'*.jpg'):
        img = cv2.imread(path_train[i]+file)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        #first attribute: histogram distance 
        hist = cv2.calcHist([img], [0, 1, 2], None, histogram_bins,[0, 256, 0, 256, 0, 256])
        feature_1_label = 0
        feature_1_dist = cv2.compareHist(hist, histogram_avg[0], cv2.HISTCMP_CHISQR_ALT)
        for j in range(1,len(path_train)):
            dist = cv2.compareHist(hist, histogram_avg[j], cv2.HISTCMP_CHISQR_ALT)
            if(dist < feature_1_dist):
                feature_1_dist = dist
                feature_1_label = j

        #second attribute: SURF keystrokes count
        surf = cv2.xfeatures2d.SURF_create(surf_thresh)
        surf.setUpright(True)
        keyPoints, _ = surf.detectAndCompute(img_gray,None)
        if(len(keyPoints)<10):
            feature_2_label = 1
        elif(len(keyPoints)>800):
            feature_2_label = 0
        else: feature_2_label = -1
                
        #third attribute: LBP
        lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
        lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2, range=(0, lbp_points + 2))
        feature_3_label = 0
        feature_3_dist = distance.cityblock(lbp_hist,lbp_histogram_samples[0])
        for j in range(1,len(lbp_histogram_samples)):
            dist = distance.cityblock(lbp_hist,lbp_histogram_samples[j])
            if(dist < feature_3_dist):
                feature_3_dist = dist
                feature_3_label = j
        
        #fourth attribute: Green pixels level
        feature_4_green_level = np.mean(cv2.inRange(img, GREEN_MIN, GREEN_MAX))
        if(feature_4_green_level==0):
            feature_4_label = 1
        elif(0<feature_4_green_level<1):
            feature_4_label = 3
        elif(feature_4_green_level>19):
            feature_4_label = 2
        else : feature_4_label = 0

        
        trainData = np.vstack((trainData,[feature_1_label,feature_2_label,
                                          feature_3_label,feature_4_label,1]))
        label = np.zeros([1, 4], dtype='int32')
        label[0][i] = 1
        trainLabel = np.vstack((trainLabel,label))
    

print('\n[INFO]Train Data creation completed!\n')
## Train Data creation completed

sig = lambda t: 1/(1+np.exp(-t))
layer_1_w = np.random.random((5,7))
layer_2_w = np.random.random((7,6))
layer_3_w = np.random.random((6,5))
layer_4_w = np.random.random((5,4))

for epoch in range(0,400):
    for x,t in zip(trainData, trainLabel):
        x = x[np.newaxis]
        
        layer_1 = sig(np.dot(x, layer_1_w))   # 1. katman [7]
        layer_2 = sig(np.dot(layer_1, layer_2_w)) # 2. katman [6]
        layer_3 = sig(np.dot(layer_2, layer_3_w)) # 3. katman [5]
        layer_4 = sig(np.dot(layer_3, layer_4_w)) # çıkış katman [4]
            
        layer_4_delta = (layer_4-t)*layer_4*(1-layer_4)
        layer_3_delta = np.dot(layer_4_delta, layer_4_w.T)*layer_3*(1-layer_3)
        layer_2_delta = np.dot(layer_3_delta, layer_3_w.T)*layer_2*(1-layer_2)
        layer_1_delta = np.dot(layer_2_delta, layer_2_w.T) * (layer_1)*(1-layer_1)
        
        layer_4_w -= np.dot(layer_3.T, layer_4_delta)
        layer_3_w -= np.dot(layer_2.T, layer_3_delta)
        layer_2_w -= np.dot(layer_1.T, layer_2_delta)
        layer_1_w -= np.dot(x.T, layer_1_delta)

print('\n[INFO]Training completed\n')
## Training completed

testData = np.empty([0, 4], dtype='int32')
testLabel = np.empty([0, 4], dtype='int32')

total_accuracy = 0
for i in range(len(path_test)):
    print('\n[INFO]Testing => Loading files: ',path_test[i])
    test_observed = np.empty([0, 1], dtype='int32')
    test_expected = np.empty([0, 1], dtype='int32')
    for file in glob.glob1(path_test[i],'*.jpg'):
        img = cv2.imread(path_test[i]+file)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        #first attribute: histogram distance 
        hist = cv2.calcHist([img], [0, 1, 2], None, histogram_bins,[0, 256, 0, 256, 0, 256])
        feature_1_label = 0
        feature_1_dist = cv2.compareHist(hist, histogram_avg[0], cv2.HISTCMP_CHISQR_ALT)
        for j in range(1,len(path_train)):
            dist = cv2.compareHist(hist, histogram_avg[j], cv2.HISTCMP_CHISQR_ALT)
            if(dist < feature_1_dist):
                feature_1_dist = dist
                feature_1_label = j
    
        #second attribute: SURF keystrokes count
        surf = cv2.xfeatures2d.SURF_create(surf_thresh)
        surf.setUpright(True)
        keyPoints, _ = surf.detectAndCompute(img_gray,None)
        if(len(keyPoints)<10):
            feature_2_label = 1
        elif(len(keyPoints)>800):
            feature_2_label = 0
        else: feature_2_label = -1

        #third attribute: LBP
        lbp = local_binary_pattern(img_gray, lbp_points, lbp_radius, lbp_method)
        lbp_hist, _ = np.histogram(lbp, normed=True, bins=lbp_points + 2,range=(0, lbp_points + 2))
        
        feature_3_label = 0
        feature_3_dist = distance.cityblock(lbp_hist,lbp_histogram_samples[0])
        for j in range(1,len(lbp_histogram_samples)):
            dist = distance.cityblock(lbp_hist,lbp_histogram_samples[j])
            if(dist < feature_3_dist):
                feature_3_dist = dist
                feature_3_label = j
        
        #fourth attribute: Green pixels level        
        feature_4_green_level = np.mean(cv2.inRange(img, GREEN_MIN, GREEN_MAX))
        if(feature_4_green_level==0):
            feature_4_label = 1
        elif(0<feature_4_green_level<1):
            feature_4_label = 3
        elif(feature_4_green_level>19):
            feature_4_label = 2
        else : feature_4_label = 0
                
        testData_test = np.array([feature_1_label,feature_2_label,
                             feature_3_label,feature_4_label,1])
        testData = np.vstack((testData,np.array([feature_1_label,feature_2_label,
                             feature_3_label,feature_4_label])))
        label = np.zeros([1, 4], dtype='int32')
        label[0][i] = 1
        testLabel = np.vstack((testLabel,label))
        
        #predicting label
        layer_1 = sig(np.dot(testData_test, layer_1_w))   # 1. katman [7]
        layer_2 = sig(np.dot(layer_1, layer_2_w)) # 2. katman [6]
        layer_3 = sig(np.dot(layer_2, layer_3_w)) # 3. katman [5]
        layer_4 = sig(np.dot(layer_3, layer_4_w)) # çıkış katman [4]
        result = np.argmax(layer_4)
        test_observed = np.vstack((test_observed,result))
        test_expected = np.vstack((test_expected,i))
        
        print('output: ', labels[result], '\t expected:', labels[i])
    total_accuracy += accuracy_score(test_expected, test_observed)
    print('\n\taccuracy : ',accuracy_score(test_expected, test_observed))
    print('\n')
total_accuracy /=4
print('\tTotal average accuracy : ',total_accuracy,'\n\n\n')



####### PART 2
keyPressed = input('\nContinue with KERAS (y/n)??\n')
if 'y' in keyPressed or 'Y' in keyPressed:
    ### KERAS
    model = Sequential()
    model.add(Dense(4,input_shape=(4,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(7,activation='sigmoid'))
    model.add(Dense(6,activation='sigmoid'))
    model.add(Dense(5,activation='sigmoid'))
    model.add(Dense(4,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    best_score = 0.0
    best_epoch = 0
    for epoch in range(5,51,5):
        model.fit(trainData[0:,0:4],trainLabel,epochs=epoch,batch_size=1,
                             validation_split=0.1,verbose=1)
        score = model.evaluate(testData,testLabel,batch_size=1,verbose=1)
        if(score[1]>best_score):
            best_epoch = epoch
            best_score = score[1]
    print('\nbest score: ', best_score,'\tbest_epoch: ',best_epoch)
    scores = np.zeros([4, ])
    
    print('Predictions with batch_size(1..4) based on best_score...')
    for t in range(1,5):
        model.fit(trainData[0:,0:4],trainLabel,epochs=best_epoch,batch_size=1,
                         validation_split=0.1,verbose=0)
        score = model.evaluate(testData,testLabel,batch_size=t,verbose=0)
        print('\nbatch_size: ',t,'\tscore: ',score[1])
        scores[t-1] = score[1]

    plt.xlabel('batch_size')
    plt.ylabel('accuracy')
    plt.axis([1, 4, 0.3, 1.0])
    plt.grid(True)
    plt.bar([1,2,3,4],scores)
    plt.show()





####### PART 3
keyPressed = input('\nContinue with CNN (y/n)??\n')
if 'y' in keyPressed or 'Y' in keyPressed:
    ### CNN
    activ = ['sigmoid','tanh','relu','PReLu']
    trainImages = []
    trainLabel = []
    for i in range(len(path_train)):
        print('\n[INFO] Loading train files: ',path_train[i])
        for file in glob.glob1(path_train[i],'*.jpg'):
            img = cv2.imread(path_train[i]+file)
            img = cv2.resize(img, (200, 200))
            trainImages.append(img)
            label = np.zeros([1, 4], dtype='int32')
            label[0][i] = 1
            label = np.array(label)
            trainLabel.append(label)

    testImages = []
    testLabel = []
    for i in range(len(path_test)):
        print('\n[INFO] Loading test files: ',path_test[i])
        for file in glob.glob1(path_test[i],'*.jpg'):
            img = cv2.imread(path_test[i]+file)
            img = cv2.resize(img, (200, 200))
            testImages.append(img)
            label = np.zeros([1, 4], dtype='int32')
            label[0][i] = 1
            testLabel.append(label)
    
    
    trainImages = np.array(trainImages)
    trainLabel = np.array(trainLabel)
    trainLabel = trainLabel[0:,0,0:]
    testImages = np.array(testImages)
    testLabel = np.array(testLabel)
    testLabel = testLabel[0:,0,0:]
    
    
    for act in range(0,4):
        start_time = time.time()
        model = Sequential()
        if(act == 3):
            model.add(Conv2D(10, (5, 5), padding='same', input_shape=[200,200,3]))
            model.add(PReLU())
            model.add(Conv2D(6, (3, 3)))
            model.add(PReLU())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.1))
            model.add(Flatten())
            model.add(Dense(4))
            model.add(PReLU())
        else:
            model.add(Conv2D(10, (5, 5), padding='same', activation=activ[act],
                         input_shape=[200,200,3]))
            model.add(Conv2D(6, (3, 3), activation = activ[act]))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.1))
        
            model.add(Flatten())
            model.add(Dense(4, activation=activ[act]))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #☺model.summary()
        print('\n\n\n[INFO] Activation func: ', activ[act])
        

        history = model.fit(trainImages, trainLabel, batch_size=50, epochs=3, verbose=1, 
                       validation_data=(testImages, testLabel))
        
        total_time = time.time()-start_time
        print('\nExec time "',activ[act],'" :', total_time, ' sec\n')
        plt.figure(figsize=[5,3])
        plt.plot(history.history['acc'],'r',linewidth=3.0)
        plt.plot(history.history['val_acc'],'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=10)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves: '+ activ[act]+'\ntime: '+ str(total_time),fontsize=16)
    