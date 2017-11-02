import numpy as np
import pandas as pd

X = pd.read_csv('InputData/ATNTFaceImages400.txt', sep=",", header=None)
df = pd.DataFrame.transpose(X);
X = np.array(df)


[m,n] = X.shape;

train_data = np.zeros((360,n))
test_Data = np.zeros((40,n))

trainIndx = -1;
testIndx = -1;


for i in range(0, m):
    if (i+1) % 10 == 0:
        testIndx = testIndx + 1
    else:
        trainIndx = trainIndx + 1;
    for j in range(0, n):
        if (i+1) % 10==0:
            test_Data[testIndx,j] = X[i,j]
        else:
            train_data[trainIndx, j] = X[i, j]


f = pd.DataFrame(train_data)

X = np.array(f.drop([0],1))
y = np.array(f[0])

y = y.transpose()

np.savetxt("ATNT400_TrainDataXY.txt",test_Data.transpose() , fmt="%d", delimiter=",")
np.savetxt("ATNT400_TestDataXY.txt",train_data.transpose() , fmt="%d", delimiter=",")

print("Training and Test Data for ATNT400 has been successfully generated!!")