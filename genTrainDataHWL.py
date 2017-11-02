import numpy as np
import pandas as pd

X = pd.read_csv('InputData/HandWrittenLetters.txt', sep=",", header=None)
df = pd.DataFrame.transpose(X);
X = np.array(df)

[m,n] = X.shape;

train_data = np.zeros((60,n))
test_Data = np.zeros((18,n))

trainIndx = -1;
testIndx = -1;


for i in range(0, m):
    if i>=0 and i<30 or i>m-1-39 and i<=m-9:
        trainIndx = trainIndx + 1;
    elif i>=30 and i<=39 or i>m-9 and i<m:
        testIndx = testIndx+1;
    for j in range(0, n):
        if i>=0 and i<30 or i>m-1-39 and i<m-9:
            train_data[trainIndx, j] = X[i, j]
        elif i >=30 and i <=39 or i >=m - 9 and i < m:
             test_Data[testIndx, j] = X[i, j]



f = pd.DataFrame(train_data)

X = np.array(f.drop([0],1))
y = np.array(f[0])

y = y.transpose()

np.savetxt("HWL_TrainDataXY.txt",train_data.transpose() , fmt="%d", delimiter=",")
np.savetxt("HWL_TestDataXY.txt",test_Data.transpose() , fmt="%d", delimiter=",")

print("2-Class Training and Test Data for HandWrittenLetter data has been successfully generated!!")