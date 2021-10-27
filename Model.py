# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:00:27 2020

@author: Harikrishnan.V
"""
#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#import dataset

df = pd.read_excel("iris.xls")
X = df.drop(["Classification"],axis=1)
y = df.iloc[:,-1].values

#Encoding categorical variables

encoder = LabelEncoder()
y = encoder.fit_transform(y)

#Training the dataset

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=17)

#implementing Knearest neighbor classifier

model = KNeighborsClassifier(n_neighbors=7,metric="minkowski")

model.fit(X_train,y_train)

#save the model
pickle.dump(model,open("Kmodel.pkl","wb"))




