import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import torchvision.transforms as transforms
batch_size=100
def getXmean(x_train):
    x_train = np.reshape(x_train, (x_train.shape[0], -1))  # Turn the image to 1-D
    mean_image = np.mean(x_train, axis=0)  # 求每一列均值。即求所有图片每一个像素上的平均值
    return mean_image


def centralized(x_test, mean_image):
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    x_test = x_test.astype(np.float)
    x_test -= mean_image  # Subtract the mean from the graph, and you get zero mean graph
    return x_test

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


#MNIST dataset
train_dataset=dsets.MNIST(root='/ml/pymnist', #数据根目录
                          train=True, #选择训练集
                          transform=None, #无预处理
                          download=True #从网上下载图片
                          )
test_dataset=dsets.MNIST(root='/ml/pymnist', #数据根目录
                          train=False, #选择测试集
                          transform=None, #无预处理
                          download=True #从网上下载图片
                          )
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)#数据打乱
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)
print("train_data:",train_dataset.train_data.size())
print("train_labels:",train_dataset.train_labels.size())
print("test_data:",test_dataset.test_data.size())
print("test_labels:",test_dataset.test_labels.size())
import matplotlib.pyplot as plt
#digit=train_loader.dataset.train_data[0]
#plt.imshow(digit,cmap=plt.cm.binary)
#plt.show()
#print(train_loader.dataset.train_labels[0])



if __name__=='__main__':
    X_train=train_loader.dataset.train_data.numpy()#转为numpy矩阵
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    y_train=train_loader.dataset.train_labels.numpy()
    X_test=test_loader.dataset.test_data[:1000].numpy()
    X_test=centralized(X_test,mean_image)
    y_test=test_loader.dataset.test_labels[:1000].numpy()
    num_test=y_test.shape[0]
    y_test_pred=kNN_classify(5,'M',X_train,y_train,X_test)
    num_correct=np.sum(y_test_pred==y_test)
    accuracy=float(num_correct)/num_test
    print('Got %d/%d correct=>accuracy: %f'%(num_correct,num_test,accuracy))

