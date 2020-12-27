import numpy as np

def generate_dataset(n,m,ch):
    X = np.random.randn(n,m)
    if ch==True:
        y=np.random.choice([-1, 1], size=(n,))
    else:
        y=np.random.randn(n,1)
    return X,y


def two_layer_loss_vectorized(w,X,y,reg):
  w=w.reshape(X.shape[1], 1)
  value = np.dot(X,w)
  loss = np.sum(np.where(value<0,y**2,(y-value)**2),axis=0)
  grad = np.sum(np.where(value<0,0,-2*(y-value)*X), axis=0)
  loss = loss +  0.5*reg*np.sum(w.T*w.T)
  grad = grad + reg*w.T
  return [loss,grad]

def two_layer_naive(w,x,y,reg):
    loss,value = 0, 0
    grad = np.zeros_like(w).T
    count,dim = x.shape
    for i in range(count):
        value = 0
        for j in range(dim):
            value = value + x[i][j]*w[j]
        h_x = np.where(value>0,value,0)
        loss += np.sum((y[i]-h_x)**2)
        grad += 2*(y[i]-h_x)*(-x[i] if value>0 else 0)
    loss+=0.5*reg*np.sum(w.T*w.T)
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
x,y = generate_dataset(n,m,ch=False)
w = np.random.randn(m,1)
import time
tic = time.time()
print("Executing Two_Layer_Loss_Naive......")
loss,grad=two_layer_naive(w,x,y,1)
print(loss)
print(grad)
print("Execution Time:",time.time()-tic)

print('--------------------')

tic = time.time()
print("Executing Two_Layer_Loss_Vectorized......")
loss1,grad1=two_layer_loss_vectorized(w,x,y,1)
print(loss1)
print(grad1)
print("Execution Time:",time.time()-tic)

funObj = lambda w: two_layer_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.000000001)
print("numerical_grad",x)
print()

print("==================================================================================================================")
print("scalablilty test")

m=10
n=100000
print("M",m)
print("N",n)
x,y = generate_dataset(n,m,ch=False)
w = np.random.randn(m,1)
import time
tic = time.time()
print("Executing Two_Layer_Loss_Naive......")
loss,grad=two_layer_naive(w,x,y,1)
print(loss)
print(grad)
print("Execution Time:",time.time()-tic)

print('--------------------')

tic = time.time()
print("Executing Two_Layer_Loss_Vectorized......")
loss1,grad1=two_layer_loss_vectorized(w,x,y,1)
print(loss1)
print(grad1)
print("Execution Time:",time.time()-tic)

funObj = lambda w: two_layer_loss_vectorized(w,x,y,1)[0]
x = numericalGrad(funObj,w,0.000000001)
print("numerical_grad",x)
print()

#Vectorized works faster