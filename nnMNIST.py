#MNIST dataset
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import torchvision.transforms as transforms

train_dataset=dsets.MNIST(root='/ml/pymnist',
                          train=True,
                          transform=transforms.ToTensor(),#转换成tensor变量
                          download=False
                          )
test_dataset=dsets.MNIST(root='/ml/pymnist',
                          train=False,
                          transform=transforms.ToTensor(),#转换成tensor变量
                          download=False
                          )

def init_network():
    network={}
    weight_scale=1e-3
    network['W1']=np.random.randn(784,50) *weight_scale
    network['b1']=np.ones(50)
    network['W2']=np.random.randn(50,100)*weight_scale
    network['b2']=np.ones(100)
    network['W3']=np.random.randn(100,10)*weight_scale
    network['b3']=np.ones(10)
    return network

def _relu(x):
    return np.maximum(0,x)

def _softmax(x):
    if x.ndim==2:
        c=np.max(x,axis=1)
        x=x.T-c#溢出对策
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    c=np.max(x)
    exp_x=np.exp(x-c)
    return exp_x/np.sum(exp_x)

def mean_squared_error(p,y):
    return np.sum((p-y)**2/y.shape[0])

def cross_entropy_error(p,y):
    delta=1e-7
    batch_size=p.shape[0]
    return -np.sum(y*np.log(p+delta))/batch_size


def forward(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=x.dot(w1)+b1
    z1=_relu(a1)
    a2=z1.dot(w2)+b2
    z2=_relu(a2)
    a3=z2.dot(w3)+b3
    y=a3
    return y

network=init_network()
accuracy_cnt=0
batch_size=100
x=test_dataset.test_data.numpy().reshape(-1,28*28)
labels=test_dataset.test_labels
finallables=labels.reshape(labels.shape[0],1)
bestloss=float('inf')
for i in range(0,len(x),batch_size):
    network=init_network()
    x_batch=x[i:i+batch_size]
    y_batch=forward(network,x_batch)
    one_hot_labels=torch.zeros(batch_size,10).scatter_(1,finallables[i:i+batch_size],1)
    loss=cross_entropy_error(one_hot_labels.numpy(),y_batch)
    if loss<bestloss:
        bestloss=loss
        bestw1,bestw2,bestw3=network['W1'],network['W2'],network['W3']
    print("best loss: is %f"%(bestloss))
a1=x.dot(bestw1)
z1=_relu(a1)
a2=z1.dot(bestw2)
z2=_relu(a2)
a3=z2.dot(bestw3)
y=_softmax(a3)
print(y)
Yte_predict=np.argmax(y,axis=1)
one_hot_labels=torch.zeros(x.shape[0],10).scatter_(1,finallables,1)
true_labels=np.argmax(one_hot_labels.numpy(),axis=1)
print(np.mean(Yte_predict==true_labels))
