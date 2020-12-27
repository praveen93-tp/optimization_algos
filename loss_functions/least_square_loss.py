import numpy as np

def generate_dataset(n,m,ch):
    X = np.random.randn(n,m)
    if ch==True:
        y=np.random.choice([-1, 1], size=(n,))
    else:
        y=np.random.randn(n,1)
    return X,y

def least_square_loss_vectorized(w,x,y,reg):
    loss=0
    grad=np.zeros_like(w).T
    f_x = np.dot(w.T,x.T)
    h_x = y.T-f_x
    #loss+=np.sum(np.power(h_x,2))
    #grad+=np.dot(2*h_x, x)
    loss+= np.sum(np.power(h_x,2))+0.5*reg*np.sum(w.T*w.T)
    grad+=np.dot(2*h_x,(-x))+reg*w.T
    return loss,grad

def least_square_loss_naive(w,x,y,reg):
    loss,value=0,0
    grad = np.zeros_like(w).T
    count,dimentions = x.shape
    for i in range(count):
        value=0
        for j in range(dimentions):
            value = value+x[i][j]*w[j]
        h_x = y[i]-value
        loss+=h_x**2
        grad+=2*h_x*(-x[i])
    loss+=0.5*reg*np.sum(w.T * w.T)
    grad+=reg*w.T
    return loss,grad



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
print("Executing Least_Square_Loss_Naive......")
loss1,grad1=least_square_loss_naive(w,x,y,1)
print("loss_naive",loss1)
print("grad_naive",grad1)
print("Execution Time:",time.time()-tic)

tic = time.time()
print("Executing Least_Square_Loss_Vectorized......")
loss,grad=least_square_loss_vectorized(w,x,y,1)
print("loss_vectorized",loss)
print("grad_vectorized",grad)
print("Execution Time:",time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: least_square_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.00001)
print("numerical_grad",x)

print("===============================================================================================================")

m=10
n=100000
print("M",m)
print("N",n)
x,y = generate_dataset(n,m,ch=True)
w = np.random.randn(m,1)
import time
tic = time.time()
print("Executing Least_Square_Loss_Naive......")
loss1,grad1=least_square_loss_naive(w,x,y,1)
print("loss_naive",loss1)
print("grad_naive",grad1)
print("Execution Time:",time.time()-tic)

tic = time.time()
print("Executing Least_Square_Loss_Vectorized......")
loss,grad=least_square_loss_vectorized(w,x,y,1)
print("loss_vectorized",loss)
print("grad_vectorized",grad)
print("Execution Time:",time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: least_square_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.00001)
print("numerical_grad",x)