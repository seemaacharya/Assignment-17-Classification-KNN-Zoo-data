# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:35:23 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Loading the dataset
data = pd.read_csv("Zoo.csv")
df = pd.DataFrame(data)
df.head()
df.shape
df.info()

##standardize the variables(as the variables with the large scale in the dataset will have the larger effect on the distance between the observations)

x= df.drop(["animal name","type"],axis=1).values
y= df["type"].values

#converting into dataframe
x1= pd.DataFrame(x)
y1= pd.DataFrame(y)

#train_test split
X_train,X_test,Y_train,Y_test= train_test_split(x1,y1,test_size=0.3,random_state=42)

#Building model using KNN 
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
pred = knn.predict(X_test)

#KNN score(to evaluate the model)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
#Here to check the accuracy consider F1 score which is 81%

