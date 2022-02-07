import numpy as np
X=np.array([[0.6,0.9]])
print(X.shape)

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

def numerial_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    while not it.finished:
        idx=it.multi_index
        tmp_val=x[idx]
        x[idx]=float(tmp_val)+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)

        x[idx]=tmp_val#还原值
        it.iternext()

    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=1000):
    x=init_x
    for i in range(step_num):
        grad=numerial_gradient(f,x)
        x-=lr*grad
    return x

class simpleNet:
    def __init__(self):
        np.random.seed(0)
        self.W=np.random.randn(2,3)

    def forward(self,x):
        return np.dot(x,self.W)

    def loss(self,x,y):
        z=self.forward(x)
        p=_softmax(z)
        loss=cross_entropy_error(p,y)
        return loss


net=simpleNet()
print(net.W)
X=np.array([[0.6,0.9]])
p=net.forward(X)
print('预测值为：',p)
print('预测的类别为：',np.argmax(p))
y=np.array([0,0,1])
f=lambda w:net.loss(X,y)
dw=gradient_descent(f,net.W)
print(dw)
print('损失值变为',cross_entropy_error(_softmax(np.dot(X,dw)),y))
print('预测类别为：',np.argmax(np.dot(X,dw)))
