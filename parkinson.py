# -*- coding: utf-8 -*-
"""
Created on Thu Mar  11 10:16:51 2021

@author: VEDANT
"""

#library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#entering the dataset

dataset = pd.read_csv('parkinsons.csv')
X = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23]].values
Y = dataset.iloc[:, 17].values

#splitting into test set and train set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# feature scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#applying PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

#fitting into knn model

from sklearn.neighbors import KNeighborsClassifier
classifi = KNeighborsClassifier(n_neighbors = 8,p=2,metric ='minkowski')
classifi.fit(X_train,Y_train)

#predicting reults
Y_pred = classifi.predict(X_test)


#fitting the model in SVM
from sklearn.svm import SVC
classifi2 = SVC()
classifi2.fit(X_train,Y_train)

#predicting reults
y2_pred = classifi2.predict(X_test)

#fitting the data in random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifi3 = RandomForestClassifier(n_estimators=16,criterion = "entropy",random_state=0)
classifi3.fit(X_train,Y_train)

#predicting reults
y3_pred = classifi3.predict(X_test)



#Analyzing
from sklearn.metrics import confusion_matrix,accuracy_score

#KNN model Score
cm=confusion_matrix(Y_test,Y_pred)*100
accuracy_score(Y_test,Y_pred)

#SVM model Score
cm2=confusion_matrix(Y_test,y2_pred)*100
accuracy_score(Y_test,y2_pred)

#Random Forest Classifier Model Score
cm3=confusion_matrix(Y_test,y3_pred)*100
accuracy_score(Y_test,y3_pred)



