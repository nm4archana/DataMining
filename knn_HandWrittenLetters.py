import numpy as np
from sklearn import neighbors
from sklearn.model_selection import cross_val_score,KFold
import pandas as pd

"""
Date   : Sep 25 2017
@author: Archana Neelipalayam Masilamani
Project Description:
Implemented K-Neighbors Classification using scikit-learn. Machine Learning in Python.

The dataset used is HandWritten Letters.

Text file: 
1st row is cluster labels. 
2nd-end rows: each column is a feature vectors (vector length=20x16).

Total 26 classes. each class has 39 images. Total 26*39=1014 images.

The task is to predict the HandWritten Letter using K-Neighbors Classification

"""

"""
Import data from the textfile
"""
x = pd.read_csv('InputData/HandWrittenLetters.txt', sep=",", header=None)

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
K-Nearest Neighbor Classification - 
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
clf = neighbors.KNeighborsClassifier(n_neighbors=5);


"""
Evaluating score by cross-validation
"""
scores = cross_val_score(clf,X,y,cv = 5)

"""
Print Classification Accuracy
"""
for m in scores:
    print("\nClassification accuracy: {:.2f}".format(m))




"""
#using K-Fold
cv = KFold(n_splits=5,shuffle=True)
i = 0;
scores = []
for train_index , test_index in cv.split(x):
    clf.fit(x[train_index], y[train_index])
    classification_accuracy = clf.score(x[test_index], y[test_index])
    scores.append(classification_accuracy)
    i += 1
print(scores)
"""

























































