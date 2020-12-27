import numpy as np


def generate_dataset(n,m,ch):
    X = np.random.randn(n,m)
    if ch==True:
        y=np.random.choice([-1, 1], size=(n,)).reshape(n,1)
    else:
        y=np.random.randn(n,1)
    return X,y

def hinge_loss_vectorized(w,X,y,reg):
    value = np.multiply(y,np.dot(X,w))
    loss = np.sum(np.where(1-value<0,0,1-value))
    loss = loss+0.5*reg*np.sum(w.T*w.T)
    grad = np.sum(np.where(value>1,0,-y*X),axis=0)+reg*w.T
    return [loss,grad]


def hinge_loss_naive(w,x,y,reg):
    loss,grad,value = 0,0,0
    for i in range(x.shape[0]):
        value = 0
        for j in range(len(x[i])):
            value = value+x[i][j]*w[j]
        temp = value*y[i]
        loss=loss+max(0,1-temp)
        grad+=0 if temp> 1 else -y[i]*x[i]
    loss=loss+0.5*reg*np.sum(w.T*w.T)
    grad = grad+reg*w.T
    return [loss,grad]

def numericalGrad(funObj,w,epsilon):
    m = len(w)
    grad = np.zeros(m)
    for i in range(m) :
        wp = np.copy(w)
        wn = np.copy(w)
        wp[i] = w[i] + epsilon
        wn[i] = w[i] - epsilon
        grad[i] = (funObj(wp)-funObj(wn))/(2*epsilon)
    return grad


m=10
n=100
print("M",m)
print("N",n)
x,y = generate_dataset(n,m,ch=True)
w = np.random.randn(m,1)
import time
tic = time.time()
print("Executing Hinge_Loss_Naive......")
loss1,grad1=hinge_loss_naive(w,x,y,1)
print("loss_naive",loss1)
print("grad_naive",grad1)
print("Execution Time:",time.time()-tic)

tic = time.time()
print("Executing Hinge_Loss_Vectorized......")
loss,grad=hinge_loss_vectorized(w,x,y,1)
print("loss_vectorized",loss)
print("grad_vectorized",grad)
print("Execution Time:",time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: hinge_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.00001)
print("numerical_grad",x)

print("=================================================================================================")
m=10
n=100000
print("M",m)
print("N",n)
x,y = generate_dataset(n,m,ch=True)
w = np.random.randn(m,1)
import time
tic = time.time()
print("Executing Hinge_Loss_Naive......")
loss1,grad1=hinge_loss_naive(w,x,y,1)
print("loss_naive",loss1)
print("grad_naive",grad1)
print("Execution Time:",time.time()-tic)

tic = time.time()
print("Executing Hinge_Loss_Vectorized......")
loss,grad=hinge_loss_vectorized(w,x,y,1)
print("loss_vectorized",loss)
print("grad_vectorized",grad)
print("Execution Time:",time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: hinge_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.00001)
print("numerical_grad",x)



exit()




