import os
import numpy as np
import sklearn.model_selection
from sklearn import svm
import utils
import concurrent.futures
import pickle 
import time
import sys
def cross_val_worker(clf, c, kernel, y_train, train_idxs, val_idxs):
    svm_clf = clf(C = c, kernel = 'precomputed', cache_size = 100)
    svm_clf.fit(kernel[train_idxs][:, train_idxs], y_train[train_idxs])
    preds = svm_clf.predict(kernel[val_idxs][:, train_idxs])
    return np.mean(preds != y_train[val_idxs])
def cross_val_svm(y_train, kernel, cs, folds = 5,):
    errs = np.zeros_like(cs)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(cross_val_worker, svm.SVC, c, kernel, y_train, train_idxs, val_idxs): (i,j)
            for i, c in enumerate(cs)
            for j, (train_idxs, val_idxs) in enumerate(sklearn.model_selection.KFold(n_splits = folds).split(range(len(y_train))))
        }
        for future in concurrent.futures.as_completed(futures):
            i,_ = futures[future]
            errs[i] += future.result()/folds
    return errs
if __name__ == '__main__':
    # init data
    _, y_train = utils.load_data(os.path.join(os.getcwd(),"data/cleaned_tweets_train.tsv"),type="train")
    y_train = np.array(y_train)
    # receive params
    alpha_low = float(sys.argv[1])
    alpha_high = float(sys.argv[2])
    n_alpha = int(sys.argv[3])
    deg_low = int(sys.argv[4])
    deg_high = int(sys.argv[5])
    c_low = float(sys.argv[6])
    c_high = float(sys.argv[7])
    n_c = int(sys.argv[8])
    folds = int(sys.argv[9])
    # init params
    alphas = np.logspace(alpha_low, alpha_high, n_alpha)
    degs = np.arange(start = deg_low, stop = deg_high + 1)
    cs = np.logspace(c_low, c_high, n_c)
    errs = np.zeros((len(degs), len(alphas), len(cs)))
    kernel_paths = {f"BoW_poly_gram/bow_gram_poly_deg_{deg}_alpha_{alpha:.0e}.pkl": (i, j) for i, deg in enumerate(degs) for j, alpha in enumerate(alphas)}
    # submit jobs
    start = time.time()
    for path in kernel_paths:
        i, j = kernel_paths[path]
        with open(path, 'rb') as f:
            kernel = pickle.load(f)
        err = cross_val_svm(y_train, kernel, folds = folds, cs = cs)
        errs[i, j] = err
        del kernel
    with open(f'BoW_poly_CV/BoW_poly_CV_alpha_{alpha_low}_{alpha_high}_n_alpha_{n_alpha}_deg_{deg_low}_{deg_high}_C_{c_low}_{c_high}_n_C_{n_c}_folds_{folds}.pkl', 'wb') as f:
        pickle.dump((degs, alphas, cs, errs), f)
    end = time.time()
    print(f"Time taken: {end - start}")