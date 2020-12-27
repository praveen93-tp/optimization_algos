from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

X = loadmat(r"../Project_2/data/train.mat")
y = np.loadtxt(r"../Project_2/data/train.targets")

X = X['X'].todense()

def LogisticLoss(w, X, y, lam):
    m = X.shape[0]
    Xw = np.dot(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)
    f = np.sum(np.logaddexp(0,-yXw)) + 0.5*lam*np.sum(np.multiply(w,w))
    gMul = 1/(1 + np.exp(yXw))
    ymul = -1*np.multiply(yT, gMul)
    g = np.dot(ymul.reshape(1,-1),X) + lam*w.reshape(1,-1)
    g = g.reshape(-1,1)
    return [f, g]

def HingeLoss(w, X, y, lam):
    # Computes the cost function for all the training samples
    Xw = np.matmul(X,w)
    yT = y.reshape(-1,1)
    yXw = np.multiply(yT,Xw)
    f = np.sum(np.maximum(0, 1 - yXw.T)) + 0.5*np.sum(w.T*w)
    ymul = -1*np.multiply(yT,np.double(1 > yXw))
    g = np.matmul(ymul.reshape(1,-1),X).reshape(-1,1)  + 1*w.reshape(-1,1)
    return [f, g]


def gradient_descent_fixed_learning_rate(func,w,max_iteration,X,y,reg):
    learning_rate = 1e-05
    values = []
    cost, dw = func(w, X, y, reg)
    for i in range(max_iteration):
        w = w - learning_rate* dw
        cost, dw = func(w, X, y, reg)
        #print(learning_rate, cost)
        values.append(cost)
    return cost,w,values



def gradient_descent_armijo_line_search(func,w,max_iteration,X,y,reg):
    f_old,g_old = func(w,X,y,reg)
    learning_rate = 1/np.linalg.norm(g_old)
    values = []
    max_iterations,gamma = 250,1e-04
    count = 0
    for i in range(max_iteration):
        #if(count>max_iterations):
            #break;
        f_old, g_old = func(w, X, y,reg)
        #print(learning_rate, f_old)
        wp = w - learning_rate * g_old
        f_curr, g_curr = func(wp,X,y,reg)
        while f_curr>f_old-gamma*learning_rate*np.dot(g_old.T, g_old):
            learning_rate = ((learning_rate**2) * np.dot(g_old.T,g_old))/ (2*(f_curr+learning_rate*np.dot(g_old.T,g_old)-f_old))
            learning_rate = learning_rate.item((0, 0))
            wp = w - learning_rate * g_old
            f_curr, g_curr = func(wp,X,y,reg)
            #count = count+1
        learning_rate = min(1, ((2*(f_old - f_curr))/(np.dot(g_old.T, g_old))))
        learning_rate = learning_rate.item((0, 0))
        f_old, g_old = f_curr, g_curr
        w = wp
        values.append(f_old)
        #count = count + 1
    return f_old,g_old,values


def accelerated_gradient_descent(func,w,max_iteration,X,y,reg):
    f_old, g_old = func(w, X, y, reg)
    learning_rate = 1/np.linalg.norm(g_old)
    gamma,lambda_prev,lambda_curr = 0,0,1
    y_prev = w
    gamma_line_search = 1e-04
    values = []
    count = 0
    for i in range(max_iteration):
        #if (count > max_iteration):
            #break;
        #print(learning_rate, f_old)
        y_curr = w - learning_rate * g_old
        wp = (1 - gamma) * y_curr + gamma * y_prev
        #wp = y_curr + gamma * (y_curr-y_prev)
        y_prev = y_curr
        lambda_tmp = lambda_curr
        lambda_curr = (1 + math.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp
        gamma = (1 - lambda_prev)/lambda_curr
        #gamma = (lambda_prev-1)/lambda_curr
        f_curr, g_curr = func(wp,X,y,reg)
        while (f_curr > f_old - gamma_line_search * learning_rate * np.dot(g_old.T, g_old)):
            learning_rate = ((learning_rate ** 2) * np.dot(g_old.T, g_old))/(2*(f_curr + learning_rate * np.dot(g_old.T, g_old) - f_old))
            learning_rate = learning_rate.item((0, 0))
            gamma = 0
            wp = w - learning_rate * g_old
            f_curr, g_curr = func(wp, X, y, reg)
            #count = count+1
        learning_rate = min(1, ((2*(f_old - f_curr))/(np.dot(g_old.T, g_old))))
        learning_rate = learning_rate.item((0, 0))
        w = wp
        f_old, g_old = f_curr, g_curr
        values.append(f_old)
        #count=count+1
    return f_old,g_old,values


def conjugate_gradient_descent(func,w,max_iteration,X,y,reg):
    f_old, g_old = func(w, X, y, 0.5)
    learning_rate = 1 / np.linalg.norm(g_old)
    gamma = 1e-04
    d_old = -g_old
    beta = 0
    x_prev = w
    values = []
    count = 0
    for i in range(max_iteration):
        #if (count > max_iteration):
            #break;
        x_curr = x_prev + learning_rate * (d_old)
        f_curr, g_curr = func(x_curr, X, y, reg)
        #print(learning_rate, f_old)
        while (f_curr > f_old-gamma*learning_rate*np.dot(g_old.T, g_old)):
            learning_rate = ((learning_rate**2) * np.dot(g_old.T,g_old))/ (2*(f_curr+learning_rate*np.dot(g_old.T,g_old)-f_old))
            learning_rate = learning_rate.item((0, 0))
            d_old = -g_old
            x_curr = x_prev + learning_rate * (d_old)
            f_curr, g_curr = func(x_curr,X,y,reg)
            #count=count+1
        learning_rate = min(1, ((2*(f_old - f_curr))/(np.dot(g_old.T, g_old))))
        learning_rate = learning_rate.item((0, 0))
        beta = np.linalg.norm(g_curr)/np.linalg.norm(g_old)
        d_new = -g_curr + beta * d_old
        x_prev = x_curr
        f_old, g_old = f_curr, g_curr
        d_old = d_new
        values.append(f_old)
        #count=count+1
    return f_old, g_old,values


def brazillia_bowrein_step_gradient_descent(func,w,max_iteration,X,y,reg):
    f_old, g_old = func(w, X, y, reg)
    learning_rate = 1/np.linalg.norm(g_old)
    values = []
    max_iterations, gamma = 250, 1e-04
    count = 0
    for i in range(max_iteration):
        #if (count > max_iteration):
            #break;
        #print(learning_rate, f_old)
        wp = w - learning_rate * g_old
        f_curr, g_curr = func(wp, X, y, reg)
        while f_curr > f_old - gamma * learning_rate * np.dot(g_old.T, g_old):
            learning_rate = ((learning_rate ** 2) * np.dot(g_old.T, g_old))/(2 * (f_curr + learning_rate * np.dot(g_old.T, g_old) - f_old))
            learning_rate = learning_rate.item((0, 0))
            wp = w - learning_rate * g_old
            f_curr, g_curr = func(wp, X, y, reg)
            #count=count+1
        #learning_rate = min(1, ((2 * (f_old - f_curr)) / (np.dot(g_old.T, g_old))))
        #learning_rate = learning_rate.item((0, 0))
        #learning_rate = np.dot((wp-w).T,(wp-w))/(np.dot((wp-w).T,(g_curr-g_old))) #version 1
        learning_rate = np.dot((wp - w).T,(g_curr-g_old)) / (np.dot((g_curr-g_old).T,(g_curr-g_old)))
        learning_rate = learning_rate.item((0, 0))
        f_old, g_old = f_curr, g_curr
        w = wp
        values.append(f_old)
        #count=count+1
    return f_old, g_old,values



y = y.reshape(y.shape[0],1)

loss_function = sys.argv[1]
if loss_function=='logistic':
    loss_function=LogisticLoss
elif loss_function=='hinge':
    loss_function=HingeLoss
else:
    print('Please enter the coorect loss function')
    exit()

#loss_function = HingeLoss
w = np.zeros(X.shape[1]).reshape(X.shape[1],1)
cost,grad = loss_function(w,X,y,1)
print('Running the algorithms on:',loss_function)
print("Initial_cost",cost)
print("Running Gradient Descent.......")
cost_gradient,w_gradient,values_gradient = gradient_descent_fixed_learning_rate(loss_function,w,250,X,y,1)
print("Final cost",cost_gradient)
print("List of costs in each iteration",values_gradient)
print('***************************************************************************************************')
print('Running Arnijo Line Search')
cost_armijo,w_armijo,values_armijo = gradient_descent_armijo_line_search(loss_function,w,250,X,y,1)
print("Final cost",cost_armijo)
print("List of costs in each iteration",values_armijo)
print('***************************************************************************************************')
print('Running Acclerated Gradient Descent with line search')
cost_acclerated,w_acclerated,values_acclerated = accelerated_gradient_descent(loss_function,w,250,X,y,1)
print("Final cost",cost_acclerated)
print("List of costs in each iteration",values_acclerated)
print('***************************************************************************************************')
print('Running Conjugate gradient descent with line search')
cost_conjugate,w_conjugate,values_conjugate = conjugate_gradient_descent(loss_function,w,250,X,y,1)
print("Final cost",cost_conjugate)
print("List of costs in each iteration",values_conjugate)
print('***************************************************************************************************')
print('Running Brazillia gradient descent with line search')
cost_brazillia,w_brazillia,values_brazillia = brazillia_bowrein_step_gradient_descent(loss_function,w,250,X,y,1)
print("Final cost",cost_brazillia)
print("List of costs in each iteration",values_brazillia)

plt.plot(values_gradient,lw=3)
plt.plot(values_armijo,lw=3)
plt.plot(values_acclerated,lw=3)
plt.plot(values_conjugate,lw=3)
plt.plot(values_brazillia,lw=3)
plt.title("Optimization Progress")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.legend(['gradient_descent', 'armijo_line_search', 'accelerated_gradient_descent', 'conjugate_gradient_descent',' brazillia_bowrein_step_gradient_descent'], loc='upper right')
plt.show()



