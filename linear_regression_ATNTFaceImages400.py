import numpy as np
from sklearn.model_selection import cross_val_score,KFold
from sklearn import linear_model
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from numpy.linalg import pinv

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

"""
Date   : Sep 25 2017
@author: Archana Neelipalayam Masilamani
Project Description:
Implemented Linear Regression Classification with scikit-learn Machine Learning for KFold in Python.

The dataset used is ATNTFaceImages400.

Text file: 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=28x23).

Total 40 classes and each class has 10 images. Total 40*10=400 images

The task is to predict the correct class of the image using Linear Regression classification

"""

"""
Import data from the textfile
"""
x = pd.read_csv('InputData/ATNTFaceImages400.txt', sep=",", header=None)

"""
Transpose the data imported
"""
df = pd.DataFrame.transpose(x);

"""
Split the training data into X and y where X has training 
samples and y has the corresponding class labels
"""
X = np.array(df.drop([0],1))
y = np.array(df[0])

"""
Function to Implement Linear Regression Classification
"""
def linearRegression(trainX,trainY,testX,testY):
    """
    Append column of 1's to the training sample
    """
    [m, n] = trainX.shape
    m = np.ones((m, 1))
    trainXa = np.c_[m, trainX]

    [m, n] = testX.shape
    m = np.ones((m, 1))
    testXa = np.c_[m, testX]

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
            if(b == trainY[i]):
                break;
        Yarr[i][count] = 1

    """
    Compute B = (X' * X)^-1 * X' * Y
    """

    k = np.matrix.transpose(trainXa)
    B1 = np.matmul(k, trainXa)

    B1inv = pinv(B1)

    B12 = np.matmul(B1inv, k)
    B = np.matmul(B12, Yarr)

    #B = np.linalg.pinv(trainXa.T.dot(trainXa)).dot(trainXa.T).dot(Yarr)

    """
    Compute F(x)= [(1,x)*B]'
    """
    F = np.matmul(testXa, B)
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
    return score


"""
Using K-Fold in sklearn to split data, and for each split call linear regression function implemented
"""
cv = KFold(n_splits=5,shuffle=True)
i = 0;
scores = []
for train_index, test_index in cv.split(X,y):
    """
    Calling function Linear Regression to classify and calculate the accuracy
    """
    classification_accuracy = linearRegression(X[train_index],y[train_index],X[test_index],y[test_index])
    scores.append(classification_accuracy)
    i += 1

"""
Print Classification Accuracy
"""
for m in scores:
    print("\nClassification accuracy: {:.2f}".format(m))



"""
#Linear Regression for regression using Scikit-Learn Machine Learning Library
reg = linear_model.LinearRegression(fit_intercept=True)
kfold = KFold(n_splits=5,shuffle=True)
#using cross_val_score
scores = cross_val_score(reg,X,y,cv=kfold,scoring='r2')
for m in scores:
    print("\nCoefficient R^2: {:.2f}".format(m))
"""




















































