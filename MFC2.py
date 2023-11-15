import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.linalg import pinv
import cv2
import os

path = os.listdir(r'C:\Users\manas\OneDrive\Desktop\mfc\Training')
classes = {'no_tumor':0, 'pituitary_tumor':1}

X = []
Y = []
for cls in classes:
    pth = 'Training/'+cls
    for j in os.listdir(pth):   #folders in dir
        img = cv2.imread(pth+'/'+j, 0) 
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)   

print(np.unique(Y))
print(pd.Series(Y).value_counts())

X.shape, X_updated.shape
plt.imshow(X[0], cmap='gray')
#X_updated = X.reshape(len(X), -1)
X_updated.shape
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)

xtrain.shape, xtest.shape
print('max value     min value')
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print('scaled max value     min value')
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

from sklearn.decomposition import PCA
print(xtrain.shape, xtest.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()
sv.fit(xtrain, ytrain)

print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))
print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))

pred = sv.predict(xtest)
misclassified=np.where(ytest!=pred)

misclassified
print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])

dec = {0:'No Tumor', 1:'Positive Tumor'}

plt.figure(figsize=(12,8))
p = os.listdir(r'Testing/')
c=1
for i in os.listdir(r'Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    img = cv2.imread(r'Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1

plt.figure(figsize=(12,8))
p = os.listdir(r'Testing/')
c=1
for i in os.listdir(r'Testing\pituitary_tumor')[:16]:
    plt.subplot(4,4,c)
    de = r'C:\Users\manas\OneDrive\Desktop\mfc\Testing\pituitary_tumor\\'
    img = cv2.imread(de[:-1]+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
plt.show() 