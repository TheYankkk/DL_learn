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
        self.layers['Affine1']=
