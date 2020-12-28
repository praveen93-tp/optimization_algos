import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


dataset = pd.read_csv('./datasets/boston_housing_data.csv')

"""
understanding the dataset
"""
#print(dataset.head())
#print(dataset.describe(include='all'))
#print(dataset.isna().sum())
#columns_with_nullval= list(dataset.columns[dataset.isnull().any()])
#print(columns_with_nullval)
#dataset=dataset.dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
#print(dataset.isna().sum())
#print(len(dataset))
#print(dataset.isna().sum())
#print(len(dataset))
dataset = dataset.apply(lambda x: x.fillna(x.mean()), axis = 0)

"""
Correlation Check.
"""
correlation_matrix = dataset.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()

"""
Feature selection based on corelation plots
"""

X = pd.DataFrame(np.c_[dataset['LSTAT'],dataset['RM']],columns=['LSTAT','RM'])
Y = dataset['MEDV']
X = (X - np.mean(X))/np.std(X)
X.insert(0,"Bias", np.ones((506,1)),True)

"""
Splitting dataset into training and testset
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=3)
Y_train = Y_train.values.reshape(len(X_train),1)


def gradient_descent(X_train,y_train,learning_rate,theta,epoch):
    N = len(y_train)
    for i in range(epoch):
        y_t = np.dot(X_train,theta)
        grad = 1/N * np.dot(X_train.T,(y_t - y_train))
        theta = theta - learning_rate*grad
    return theta

def stochastic_gradient_descent(X_train,y_train,learning_rate,theta,epoch):
    for i in range(epoch):
        y_t = np.dot(X_train, theta)
        temp = (y_t - y_train)
        for i in range(len(X_train)):
            theta = theta - (learning_rate * (X_train.iloc[i] * temp[i])).values.reshape((3, 1))
    return theta

def gradient_descent_momentum(X_train,y_train,learning_rate,theta,epoch,eta = 0.9):
    V = np.zeros((3,1))
    for i in range(epoch):
        y_t = np.dot(X_train, theta)
        temp = (y_t - y_train)
        for j in range(len(X_train)):
            V = eta * V + (learning_rate * (X_train.iloc[j] * temp[j])).values.reshape((3, 1))
        theta = theta - V
    return theta

def nesterov_gradient_descent_momentum(X_train,y_train,learning_rate,theta,epoch,eta = 0.9):
    V = np.zeros((3,1))
    for i in range(epoch):
        theta = theta + eta * V
        y_t = np.dot(X_train, theta)
        temp = (y_t - y_train)
        for j in range(len(X_train)):
            V = eta * V - (learning_rate * (X_train.iloc[j] * temp[j])).values.reshape((3, 1))
            theta = theta + V
    return theta

def  adagrad(X_train,y_train,learning_rate,theta,epoch,eta = 0.9,zeta = 0.01):
    N = len(y_train)
    for i in range(epoch):
        y_t = np.dot(X_train, theta)
        grad = 1 / N * np.dot(X_train.T, (y_t - y_train))
        alpha_t = learning_rate / (np.sqrt(zeta + np.square(grad)))
        v = -alpha_t * grad
        theta = theta + v
    return theta



def print_metrics(X_train,Y_train,X_test,Y_test,theta):
    y_predict1 = X_train.apply(lambda x: np.dot(x, theta), axis=1)
    print("Training RMSE: " + "{:2f}".format(float(np.sqrt(mean_squared_error(Y_train, y_predict1)))))
    y_predict2 = X_test.apply(lambda x: np.dot(x, theta), axis=1)
    print("Testing RMSE: " + "{:2f}".format(float(np.sqrt(mean_squared_error(Y_test, y_predict2)))))


print("Running gradient descent.......")
theta = np.zeros((3,1))
learning_rate = 0.01
epoch=100
theta = gradient_descent(X_train,Y_train,learning_rate,theta,epoch)
print_metrics(X_train,Y_train,X_test,Y_test,theta)
print()

print("Running stochastic gradient descent.......")
theta = np.zeros((3,1))
learning_rate = 0.001
epoch=100
theta = stochastic_gradient_descent(X_train,Y_train,learning_rate,theta,epoch)
print_metrics(X_train,Y_train,X_test,Y_test,theta)
print()
print("Running gradient descent with momentum.......")
theta = np.zeros((3,1))
learning_rate = 0.01
epoch=100
eta = 0.9
theta = gradient_descent_momentum(X_train,Y_train,learning_rate,theta,epoch)
print_metrics(X_train,Y_train,X_test,Y_test,theta)
print()

print("Running gradient descent with nesterov momentum.......")
theta = np.zeros((3,1))
learning_rate = 0.0001
epoch=100
eta = 0.9
theta = nesterov_gradient_descent_momentum(X_train,Y_train,learning_rate,theta,epoch)
print_metrics(X_train,Y_train,X_test,Y_test,theta)

print()
print("Running adagrad.......")
theta = np.zeros((3,1))
learning_rate = 0.1
epoch=100
eta = 0.9
zeta = 0.01
theta = adagrad(X_train,Y_train,learning_rate,theta,epoch,zeta)
print_metrics(X_train,Y_train,X_test,Y_test,theta)
