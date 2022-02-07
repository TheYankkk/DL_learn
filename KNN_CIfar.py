import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
batch_size=100
#Cifar10
train_dataset=dsets.CIFAR10(root='/ml/pycifar',
                            train=True,
                            download=True)
test_dataset=dsets.CIFAR10(root='/ml/pycifar',
                            train=False,
                            download=True)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
digit= train_loader.dataset.data[0]

import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
print(classes[train_loader.dataset.targets[0]])

import numpy as np
def kNN_classify(k,dis,X_train,x_train,Y_test):
    assert dis=='E' or dis=='M'
    num_test=Y_test.shape[0]
    labellist=[]
    if(dis=='E'):
        for i in range(num_test):
            #欧氏距离
            distances=np.sqrt(np.sum(((X_train-np.tile(Y_test[i],(X_train.shape[0],1)))**2),axis=1))

            nearest_k=np.argsort(distances)
            topK=nearest_k[:k]
            classCount={}
            for i in topK:
                classCount[x_train[i]]=classCount.get(x_train[i],0)+1
            sortedClassCount=sorted(classCount.items(),reverse=True)
            labellist.append(sortedClassCount[0][0])
    elif (dis == 'M'):
        for i in range(num_test):
            # manhattan
            distances = np.sum((X_train - np.tile(Y_test[i], (X_train.shape[0], 1))), axis=1)
            nearest_k = np.argsort(distances)
            topK = nearest_k[:k]
            classCount = {}
            for i in topK:
                classCount[x_train[i]] = classCount.get(x_train[i], 0) + 1
            sortedClassCount = sorted(classCount.items(), reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

def getXmean(X_train):
    X_train=np.reshape(X_train,(X_train.shape[0],-1))
    mean_image=np.mean(X_train,axis=0)
    return  mean_image
def centralized(X_test,mean_image):
    X_test=np.reshape(X_test,(X_test.shape[0],-1))
    X_test=X_test.astype(np.float)
    X_test-=mean_image
    return X_test

X_train=train_loader.dataset.data
mean_image=getXmean(X_train)
X_train=centralized(X_train,mean_image)
y_train=train_loader.dataset.targets
X_test=test_loader.dataset.data[:100]
X_test=centralized(X_test,mean_image)
y_test=test_loader.dataset.targets[:100]
num_test=len(y_test)
y_test_pred=kNN_classify(6,'M',X_train,y_train,X_test)
num_correct=np.sum(y_test_pred==y_test)
accuracy=float(num_correct)/num_test
print('Got %d/%d correct=>accuracy: %f' % (num_correct,num_test,accuracy))

