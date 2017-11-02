import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd

'''
Date   : Sep 25 2017
@author: Archana Neelipalayam Masilamani
Project Description:
Implemented Support Vector Machine using scikit-learn. Machine Learning in Python.

The dataset used is ATNTFaceImages400.

Text file: 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=28x23).

Total 40 classes and each class has 10 images. Total 40*10=400 images

The task is to predict the correct class of the image using Support Vector Machine classification

'''

'''
Import data from the textfile
'''
x = pd.read_csv('InputData/ATNTFaceImages400.txt', sep=",", header=None)

'''
Transpose the data imported
'''
df = pd.DataFrame.transpose(x);

'''
Split the training data into X and y where X has training 
samples and y has the corresponding class labels
'''
X = np.array(df.drop([0],1))
y = np.array(df[0])

'''
Linear Support Vector Classification - 
http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
'''
clf = svm.LinearSVC();

'''
Evaluating score by cross-validation
'''
scores = cross_val_score(clf,X,y,cv = 5)

'''
Print Classification Accuracy
'''
for m in scores:
    print("\nClassification accuracy: {:.2f}".format(m))




