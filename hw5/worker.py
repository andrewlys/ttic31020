import os
import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn import svm
import matplotlib.pyplot as plt
import utils
from utils import TrainAndTestData

cal  = sklearn.datasets.fetch_california_housing()
seed = 0
np.random.seed(seed)

m = 3000
perm = np.random.permutation(len(cal.target))
train_i = perm[:m]
test_i = perm[m:]
train_X = cal.data[train_i,:]
train_y = cal.target[train_i]
test_X = cal.data[test_i,:]
test_y = cal.target[test_i]

housing_data = TrainAndTestData(train_X, train_y, test_X, test_y)
def RBF_kernel(beta = 1):
    def RBF_kernel_beta(x1,x2):
        return np.exp(- beta*(np.sum(x1*x1, 1)[:,np.newaxis] + np.sum(x2*x2, 1)-2*x1@x2.T ))
    return RBF_kernel_beta
def train_kernel_ridge(kernel, lmbd, x, y):
    from numpy.linalg import lstsq
    K = kernel(x,x)
    #### TASK 2 CODE
    A = (2/m) * K @ K + lmbd * K
    b = (2/m) * K @ y
    least_squares_soln = np.linalg.lstsq(A, b)[0]
    #### TASK 2 CODE
    return least_squares_soln
def predict_kernel_ridge(kernel, x, alpha, train_x):
    #### TASK 2 CODE
    return kernel(x, train_x) @ alpha
    #### TASK 2 CODE
def mean_squared_error(pred, y):
    return np.mean((pred-y)**2)
beta=0.0351
kernel = RBF_kernel(beta = beta)
def train_val_split(lmbd, train_index, test_index):
    train_X, test_X = housing_data.X_train[train_index], housing_data.X_train[test_index]
    train_y, test_y = housing_data.y_train[train_index], housing_data.y_train[test_index]
    alpha = train_kernel_ridge(kernel, lmbd, train_X, train_y)
    preds = predict_kernel_ridge(kernel, test_X, alpha, train_X)
    return mean_squared_error(preds, test_y)