import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict

class Relu:
    def __init__(self):
        self.x=None

    def forward(self,x):
        self.x=np.maximum(0,x)
        out=self.x
        return out
    def backward(self,dout):
        dx=dout
        dx[self.x<=0]=0
        return dx

class _sigmoid:
    def __int__(self):
        self.out=None

    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out

    def backward(self,dout):
        dx=dout*self.out*(1-self.out)
        return dx

class Affine:
    def __int__(self,W,b):
        self.W=W
        self.b=b
        self.x=None

def _softmax(x):
    if x.ndim==2:
        c=np.max(x,axis=1)
        x=x.T-c
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    c=np.max(x)
    exp_x=np.exp(x-c)
    return exp_x/np.sum(exp_x)

def cross_entropy_error(p,y):
    delta=1e-7
    batch_size=p.shape[0]
    return -np.sum(y*np.log(p+delta))/batch_size

class SoftmaxWithLoss:
    def __int__(self):
        self.loss=None
        self.p=None#output of softmax
        self.y=None#one hot vector

    def forward(self,x,y):
        self.y=y
        self.p=_softmax(x)
        self.loss=cross_entropy_error(self.p,self.y)
        return self.loss

    def backward(self,dout=1):
        batch_size=self.y.shape[0]
        dx=(self.p-self.y)/batch_size

        return dx


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


        #生成层
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.lastLayer=SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return  x

    def loss(self,x,y):
        p=self.predict(x)
        return self.lastLayer.forward(p,y)

    def accuracy(self,x,y):
        p=self.predict(x)
        p=np.argmax(y,axis=1)

