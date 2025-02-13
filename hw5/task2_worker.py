import os
import numpy as np
import sklearn.datasets
from sklearn.model_selection import KFold
import utils
import pickle 
from utils import TrainAndTestData
import concurrent.futures

def train_val_split(data, train, predict, loss, kernel, train_index, val_index, lmbd = 1, beta = 0.2):
    ker = kernel(beta = beta)
    train_X, val_X = data.X_train[train_index], data.X_train[val_index]
    train_y, val_y = data.y_train[train_index], data.y_train[val_index]
    alpha = train(ker, lmbd, train_X, train_y)
    preds = predict(ker, val_X, alpha, train_X)
    return loss(preds, val_y)

def RBF_kernel(beta = 1):
    def RBF_kernel_beta(x1,x2):
        return np.exp(- beta*(np.sum(x1*x1, 1)[:,np.newaxis] + np.sum(x2*x2, 1)-2*x1@x2.T ))
    return RBF_kernel_beta

def train_kernel_ridge(kernel, lmbd, x, y):
    from numpy.linalg import lstsq
    K = kernel(x,x)
    #### TASK 2 CODE
    m = x.shape[0]
    A = (K +  lmbd * np.eye(m))
    least_squares_soln = lstsq(A, y)[0]
    #### TASK 2 CODE
    return least_squares_soln
def predict_kernel_ridge(kernel, x, alpha, train_x):
    #### TASK 2 CODE
    return kernel(x, train_x) @ alpha
    #### TASK 2 CODE
def mean_squared_error(pred, y):
    return np.mean((pred-y)**2)

if __name__ == '__main__':
    # init data
    cal = sklearn.datasets.fetch_california_housing()
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
    # init param space
    betas = 3.5938136638046257e-06 + np.logspace(-6.5, -5.5, 3)
    lmbds = np.logspace(-4,-2, 9)
    folds = 5
    mses = np.zeros((len(betas), len(lmbds)))

    # parallelize and cv
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(train_val_split, housing_data, train_kernel_ridge, predict_kernel_ridge,
                            mean_squared_error, RBF_kernel, train_idx, val_idx, lmbd, beta) : (i, j)
                            for i, beta in enumerate(betas)
                            for j, lmbd in enumerate(lmbds)
                            for train_idx, val_idx in KFold(n_splits=folds).split(range(len(housing_data.X_train)))
        }
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            try:
                mses[i, j] += future.result()/folds
            except Exception as exc:
                print(f"generated an exception: {exc}")
    
    # Save results to pickle
    with open('hw5_task_2_cv_err.pkl', 'wb') as f:
        pickle.dump((betas, lmbds, mses), f)
