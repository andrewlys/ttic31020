import os
import numpy as np
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn import svm
import matplotlib.pyplot as plt
import utils
import concurrent.futures
import pickle 
import sys
import time

def cross_val_worker(clf, kernel, y_train, train_idxs, val_idxs):
    clf.fit(kernel[train_idxs][:, train_idxs], y_train[train_idxs])
    preds = clf.predict(kernel[val_idxs][:, train_idxs])
    return np.mean(preds != y_train[val_idxs])
def cross_val_svm(folds, y_train, kernel, cs, beta=1):
    err = np.zeros_like(cs)
    svms = [svm.SVC(C=c, kernel = 'precomputed', cache_size=100) for c in cs]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(cross_val_worker, svm_clf, kernel ** beta, y_train, train_idxs, val_idxs): (i,j) 
            for i, svm_clf in enumerate(svms)
            for j, (train_idxs, val_idxs) in enumerate(sklearn.model_selection.KFold(n_splits = folds).split(range(len(y_train))))
        }
        for future in concurrent.futures.as_completed(futures):
            i, _ = futures[future]
            err[i] += future.result()/folds
    return err
if __name__ == "__main__":
    # init data
    _, y_train = utils.load_data(os.path.join(os.getcwd(),"data/cleaned_tweets_train.tsv"),type="train")
    y_train = np.array(y_train)
    # fetch parameters
    beta_low = float(sys.argv[1])
    beta_high = float(sys.argv[2])
    c_low = float(sys.argv[3])
    c_high = float(sys.argv[4])
    n = int(sys.argv[5])
    folds = int(sys.argv[6])
    # init parameters
    betas = np.logspace(beta_low, beta_high, num=n)
    cs = np.logspace(c_low, c_high, num = n)
    errs = np.zeros((n, n))
    # train and evaluate
    start = time.time()
    for i, beta in enumerate(betas):
        with open(f'BoW_G_RBF_gram/BoW_G_RBF_gram_beta_{beta:.1e}.pkl', 'rb') as f:
            BoW_G_RBF_gram = pickle.load(f)
        err = cross_val_svm(folds, y_train, BoW_G_RBF_gram, cs, beta = beta)
        errs[i] = err
    with open(f'BoW_G_RBF_CV/BoW_G_RBF_CV_beta_{beta_low}_{beta_high}_C_{c_low}_{c_high}_n_{n}_folds_{folds}.pkl', 'wb') as f:
        pickle.dump((cs, betas, errs), f)
    end = time.time()
    print(f'Time taken: {end - start}')