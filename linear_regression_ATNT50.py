import numpy as np
from sklearn import linear_model
from numpy.linalg import pinv
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

"""
Date   : Sep 25 2017
@author: Archana Neelipalayam Masilamani
Project Description:
Implemented Linear Regression Classification with scikit-learn Machine Learning for KFold in Python.

The dataset used is ATNT50.

Training Data: trainDataXY.txt   
It contains 45 images. image 1-9 from class 1. image 10-18 from class 2. etc.
Each image is a column. 1st row are class labels.

Test Data: testDataXY.txt     
It contain 5 images. 
Each image is a column. 1st row are class labels.

The task is to predict the correct class of the image using Linear Regression classification
"""

"""
Import data from the textfile
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
Append column of 1's to the training sample
"""
[m,n] = trainX.shape
m = np.ones((m,1))
trainXa = np.c_[m,trainX]

[m,n] = testX.shape
m = np.ones((m,1))
testXa = np.c_[m,testX]

"""
Define an Indicator Matrix
"""

[xrow, yrow] = trainXa.shape
[xcrow] = np.unique(trainY).shape
uniqueArr = np.unique(trainY)
k = int(xrow)
l = int(xcrow)
Yarr = np.zeros(shape=(k, l))

for i in range(len(Yarr)):
        count = -1;
        for b in uniqueArr:
                count = count + 1;
                if (b == trainY[i]):
                        break;
        Yarr[i][count] = 1
"""
Compute B = (X' * X)^-1 * X' * Y
"""
k = np.matrix.transpose(trainXa)
B1 = np.matmul(k,trainXa)

B1inv = pinv(B1)

B12 = np.matmul(B1inv,k)
B = np.matmul(B12,Yarr)

#B = np.linalg.inv(trainXa.T.dot(trainXa)).dot(trainXa.T).dot(Yarr)

"""
Compute F(x)= [(1,x)*B]'
"""
F = np.matmul(testXa,B)
Fx = (np.transpose(F))

"""
To Identify the largest component of f(x) and classify
accordingly
"""
row_index = Fx.argmax(axis=0)
result = uniqueArr[row_index]

"""
Calculate the accuracy
"""
score = accuracy_score(testY, result)

"""
Print Classification Accuracy
"""
print("Test set predictions:\n{}".format(result))
print("\nClassification accuracy: {:.2f}".format(score))



"""
#Linear Regression for regression using Scikit-Learn Machine Learning Library
 
reg = linear_model.LinearRegression(fit_intercept=True)

reg.fit(trainX,trainY)

classification_accuracy = reg.score(testX,testY)
#print(reg.predict(testX))

#http://benalexkeen.com/linear-regression-in-python-using-scikit-learn/
#A common method of measuring the accuracy of regression models is to use the R2R2 statistic.

y_predict = reg.predict(testX)
mse =  mean_squared_error(y_predict, testY)
print("Coefficient R^2 : {:.2f}".format(classification_accuracy))
print("Mean Squared error: {:.2f}".format(mse))

"""


