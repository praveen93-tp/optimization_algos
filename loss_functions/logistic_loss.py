import numpy as np

def generate_dataset(n,m,ch):
    X = np.random.randn(n,m)
    if ch==True:
        y=np.random.choice([-1, 1], size=(n,)).reshape(n,1)
    else:
        y=np.random.randn(n,1)
    return X,y

def logistic_loss_naive(w,X,y,reg):
  loss,grad=0,0
  for i in range(X.shape[0]):
    value=0
    for j in range(len(X[i])):
      value = value + X[i][j]*w[j]
    f_x = value*y[i]
    loss += np.logaddexp(0,-f_x)
    grad += -(y[i]*X[i])/(1+np.exp(f_x))
  loss = loss + 0.5*reg*np.sum(w.T*w.T)
  grad = grad + reg*w.T
  return loss,grad

def logistic_loss_vectorized(w,X,y,reg):
  w,grad=w.reshape(X.shape[1],1),np.zeros(X.shape[1])
  value = np.dot(X,w)*y
  loss = np.sum(np.logaddexp(0,-value))
  f_x = y*X
  v = 1+np.exp(value)
  grad = -np.sum(np.divide(f_x,v),axis=0)
  loss = loss + 0.5*reg*np.sum(w.T*w.T)
  grad = grad + reg*w.T
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
w = np.random.randn(m)
import time
tic = time.time()
loss_naive,grad_naive = logistic_loss_naive(w,x,y,1)
print(loss_naive)
print(grad_naive)
print("Logistic Naive Time Taken : ", time.time()-tic )

tic = time.time()
loss_vec,grad_vec = logistic_loss_vectorized(w,x,y,1)
print(loss_vec)
print(grad_vec)
print("Logistic Vector Time Taken : ", time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: logistic_loss_vectorized(w,x,y,1)[0]
print(funObj(w))
print(numericalGrad(funObj,w,0.0001))
print('==============================================================================================================================')

m=10
n=100000
print("M",m)
print("N",n)
x,y = generate_dataset(n,m,ch=True)
w = np.random.randn(m)
import time
tic = time.time()
loss_naive,grad_naive = logistic_loss_naive(w,x,y,1)
print(loss_naive)
print(grad_naive)
print("Logistic Naive Time Taken : ", time.time()-tic )

tic = time.time()
loss_vec,grad_vec = logistic_loss_vectorized(w,x,y,1)
print(loss_vec)
print(grad_vec)
print("Logistic Vector Time Taken : ", time.time()-tic)

print("Executing Numerical Gradient Checking......")
funObj = lambda w: logistic_loss_vectorized(w,x,y,1)[0]
print(funObj(w))
print(numericalGrad(funObj,w,0.0001))