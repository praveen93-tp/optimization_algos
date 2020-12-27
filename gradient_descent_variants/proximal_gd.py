from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


X = loadmat(r"../Project_2/data/train.mat")
y = np.loadtxt(r"../Project_2/data/train.targets")
X = X['X'].todense()

def LogisticLoss_L1(w, X, y, lam):
	m = X.shape[0]
	Xw = np.dot(X,w)
	yT = y.reshape(-1,1)
	yXw = np.multiply(yT,Xw)
	loss = np.sum(np.logaddexp(0,-yXw))+lam*np.linalg.norm(w,1)
	gMul = 1/(1 + np.exp(yXw))
	ymul = -1*np.multiply(yT, gMul)
	grad = np.dot(ymul.reshape(1,-1),X)+lam*np.sign(w).reshape(1,-1)
	grad = grad.reshape(-1,1)
	return [loss, grad]

def GradientDescent(lossFunction,max_iterations,learning_rate,X, y,lam):
	iteration = 0
	values = []
	w = np.zeros((np.shape(X)[1], 1))
	while True:
		if iteration > max_iterations:
			break
		loss, grad = lossFunction(w, X, y, lam)
		#print(str(iteration) + ". loss: " + str(loss))
		w = w - learning_rate * grad
		values.append(loss)
		iteration += 1
	return loss,values

def prox(x,learning_rate,lam):
	val = np.maximum(np.abs(x) - learning_rate * lam, 0)
	return np.multiply(np.sign(x), val)

def ProximalGradientDescent(lossFunction, max_iterations, learning_rate, X, y, lam):
	iteration = 0
	values = []
	w = np.zeros((np.shape(X)[1], 1))
	while True:
		if iteration > max_iterations:
			break
		loss, grad = lossFunction(w, X, y, lam)
		#print(str(iteration) + ". loss: " + str(loss))
		w = prox(w - learning_rate * grad,learning_rate,lam)
		values.append(loss)
		iteration += 1
	return loss,values
import sys
#lam = sys.argv[1]
max_iterations,learning_rate,lam = 250,1e-06,1000
w = np.zeros((np.shape(X)[1], 1))
cost,grad= LogisticLoss_L1(w, X, y, lam)
print("Initial Cost :",cost)
print("Gradient Descent.....")
gradient_descent_loss,grad_values = GradientDescent(LogisticLoss_L1,max_iterations,learning_rate,X,y,lam)
print('Final Cost:',gradient_descent_loss)
print('List of costs in each iteration',grad_values)
print('***************************************************************************************************')
print("Proximal Gradient Descent.....")
prox_descent_loss,prox_values = ProximalGradientDescent(LogisticLoss_L1,max_iterations,learning_rate,X,y,lam)
print('Final Cost:',prox_descent_loss)
print('List of costs in each iteration',prox_values)

plt.plot(grad_values,lw=2)
plt.plot(prox_values,lw=2)
plt.title("Optimization Progress")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.legend(['Gradient Descent', 'Proximal Gradient Descent'])
plt.show()