import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pandas as pd


"""
Date   : Sep 25 2017
@author: Archana Neelipalayam Masilamani
Project Description:
Implemented Centroid Classification using scikit-learn. Machine Learning in Python.

The dataset used is ATNT50.

Training Data: trainDataXY.txt   
It contains 45 images. image 1-9 from class 1. image 10-18 from class 2. etc.
Each image is a column. 1st row are class labels.

Test Data: testDataXY.txt     
It contain 5 images. 
Each image is a column. 1st row are class labels.

The task is to predict the correct class of the image using Centroid Classification 

"""

"""
Import training data  and testing data from the textfile
"""
x = pd.read_csv('InputData/ATNT50/testDataXY.txt', sep=",", header=None)
y = pd.read_csv('InputData/ATNT50/trainDataXY.txt', sep=",", header=None)

"""
Transpose the training and test data imported
"""
df_test = pd.DataFrame.transpose(x)
df_train = pd.DataFrame.transpose(y)

"""
Split the training data into trainX and trainY where trainX has training 
samples and trainY has the corresponding class labels
"""
trainX = np.array(df_train.drop([0],1))
trainY = np.array(df_train[0])

"""
Split the test data into testX and testY where testX has test 
samples and testY has the corresponding class labels
"""
testX = np.array(df_test.drop([0],1))
testY = np.array(df_test[0])

"""
Nearest - Centrod Classification - 
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
"""
clf = NearestCentroid()
clf.fit(trainX,trainY)

print("Test set predictions:\n{}".format(clf.predict(testX)))
print("\nClassification accuracy: {:.2f}".format(clf.score(testX,testY)))
