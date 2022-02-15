import os
import matplotlib.pyplot as plt
import spectral
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import cv2
from peanut_class import peanut
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import pandas as pd
from keras.models import Sequential, load_model
from keras.utils import np_utils
from sklearn.decomposition import PCA
#%%

for i in range(25):
    filename =f'C:/Users/User/Desktop/CNN/penut20220210/g/{i+1}_New-1'  
    img_good = spectral.envi.open(filename + '.hdr').asarray()
    img_good = img_good[0:200,:,:]
    
for i in range(25):
    filename =f'C:/Users/User/Desktop/CNN/penut20220210/b/{i+1}_New-1'  
    img_bad = spectral.envi.open(filename + '.hdr').asarray()
    img_bad = img_good[0:200,:,:]

img = spectral.envi.open('1_New-1.hdr')
#false_color_bands = np.array(img.metadata['default bands'], 'i')* -1
#data = (img.asarray()[..., false_color_bands])
data = img.asarray()
data = data[0:200,:,:]
roi_result = []
xinput = []

plt.Figure()
plt.imshow(data[:,:,0])
temp = plt.ginput(7,show_clicks=True)
plt.show()
y = np.array([1,1,1,1,0,0,0])
data_roi = peanut()
pca = PCA(n_components=3)


arr_ans = []

for i in range(len(temp)):
    xinput.append(data[int(temp[i][1]),int(temp[i][0]),:])


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(xinput, y)  

for i in range(25):

    filename =f'C:/Users/User/Desktop/CNN/penut20220210/g/{i+1}_New-1'  
    img_good = spectral.envi.open(filename + '.hdr').asarray()
    img_good = img_good[0:200,:,:]
    arr = clf.predict(img_good.reshape([-1,img_good.shape[2]]))
    arr = np.reshape(arr,(200,409))
    peanut_roi = data_roi.test_peanut_roi(img_good, arr)
    for j in range(len(peanut_roi)):
        peanut_roi[j] = cv2.resize(peanut_roi[j], (224,224)) 
        a = pca.fit_transform(np.reshape(peanut_roi[j],(peanut_roi[j].shape[0]*peanut_roi[j].shape[1],-1)))
        a= np.reshape(a,(224,224,3))
        roi_result.append(a)
        arr_ans.append(1)

for i in range(25):
    filename =f'C:/Users/User/Desktop/CNN/penut20220210/b/{i+1}_New-1'  
    img_bad = spectral.envi.open(filename + '.hdr').asarray()
    img_bad = img_bad[0:200,:,:]
    arr = clf.predict(img_bad.reshape([-1,img_bad.shape[2]]))
    arr = np.reshape(arr,(200,409))
    peanut_roi = data_roi.test_peanut_roi(img_bad, arr)
    for j in range(len(peanut_roi)):
        peanut_roi[j] = cv2.resize(peanut_roi[j], (224,224))
        a = pca.fit_transform(np.reshape(peanut_roi[j],(peanut_roi[j].shape[0]*peanut_roi[j].shape[1],-1)))
        a= np.reshape(a,(224,224,3))
        roi_result.append(a)
        arr_ans.append(0)
        
        



'''save npy'''
arr_ans = np.array(arr_ans)
np.save('ans.npy', arr_ans)
np.save('roi.npy',roi_result)

 
   


'''ROI'''


'''data_roi = peanut()
peanut_roi = data_roi.test_peanut_roi(data, arr)
peanut_roi = np.array(peanut_roi)

print(peanut_roi)
plt.imshow(peanut_roi[0][:,:,0])'''





#%%
ans = np.load('ans.npy',allow_pickle=True)
roi = np.load('roi.npy',allow_pickle=True)


ans = np_utils.to_categorical(ans)
roi_train, roi_test, ans_train, ans_test = train_test_split( roi, ans, test_size=0.4, random_state=45)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, input_shape=(224, 224, 3), activation='relu', padding='same',strides =[2,2]))
model.add(Conv2D(filters=64, kernel_size=3,  activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=2))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(2, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(roi_train, ans_train, epochs=100, batch_size=64, verbose=1)

loss, accuracy = model.evaluate(roi_test, ans_test)
print('Test:')
print('Loss:', loss)
print('Accuracy:', accuracy)

#%%





