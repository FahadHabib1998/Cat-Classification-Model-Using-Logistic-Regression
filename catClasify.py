import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

def loadData():
    trainData = h5py.File('train_catvnoncat.h5', 'r')
    trainX = np.array(trainData["train_set_x"][:])
    trainY = np.array(trainData["train_set_y"][:])

    testData = h5py.File('test_catvnoncat.h5', 'r')
    testX = np.array(testData["test_set_x"][:])
    testY = np.array(testData["test_set_y"][:])

    classes = np.array(testData["list_classes"][:])
    
    trainY = trainY.reshape((1,trainY.shape[0]))
    testY = testY.reshape((1,testY.shape[0]))

    return trainX, testX, trainY, testY

def reshapeData(trainX,testX):
    trainNew = (trainX.reshape(trainX.shape[1]*trainX.shape[2]*trainX.shape[3],trainX.shape[0]))/255
    testNew = (testX.reshape(testX.shape[1]*testX.shape[2]*testX.shape[3],testX.shape[0]))/255
    return trainNew, testNew

def activation(z):
    return 1/(1+np.exp(-z))

def gradientDescent(w,b, X, Y, epoch, alpha):
    costarr = []
    for i in range(epoch):
        m = X.shape[1]
        
        # FORWARD PROPAGATION
        A = activation(np.dot(w.T,X)+b)
        cost = (-1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

        # BACKWARD PROPAGATION
        dw = 1/m*np.dot(X,(A-Y).T)
        db = 1/m*np.sum(A-Y)

        cost = np.squeeze(cost)

        w = w - alpha*dw
        b = b - alpha*db

        if i % 100 == 0:
            costarr.append(cost)
    return w,b,dw,db,costarr

def prediction(w,b,X):
    
    m = X.shape[1]
    predict = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = activation(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        if(A[0][i] <= 0.5):
            predict[0][i] = 0
        else:
            predict[0][i] = 1
    return predict


    
    
    
    
trainX, testX, trainY, testY = loadData()
numTrain = trainX.shape[0]
numTest = testX.shape[0]
trainX,testX = reshapeData(trainX,testX)

w = np.zeros((trainX.shape[0],1))
b = 0

w,b,dw,db,costarr = gradientDescent(w,b,trainX,trainY,2000,0.5)

test_predict = prediction(w,b,testX)
train_predict = prediction(w,b,trainX)
print(trainX.shape[0])

print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict - trainY)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict - testY)) * 100))
    
    

