import os
import numpy as np
import sklearn
from sklearn import svm
import matplotlib.pyplot as plt
import utils
import concurrent.futures
import pickle 
import sys
import time 

def BoW_inner(s1,s2):
    "returns inner product between bag-of-word feature vectors of the two input strings"
    from collections import Counter
    d1 = Counter(s1.split())
    return sum(d1[w] for w in s2.split())
def gram_matrix_worker(kernel, inner, beta, x1, x2):
    ker = kernel(inner, beta=beta)
    return ker(x1, x2)
def gram_matrix(K):
    def gram_matrix_K(xs_1, xs_2):
        return np.array([[K(x1, x2) for x2 in xs_2] for x1 in xs_1])
    return gram_matrix_K
def rbf_kernel_gram(inner, beta=1):
    """Gaussian RBF kernel.

    Returns a functoin gram(xs_1,xs_2) that calculate the (cross) gram matrix G[i,j]=K(xs_1[i],xs_2[j]])
    where K is the Gaussian RBF on the features phi, specified through the inner product in phi space."""
    def rbf_kernel_sigma_inner(xs_1,xs_2):
        return np.exp(-beta*(np.array([inner(x1, x1) for x1 in xs_1])[:, np.newaxis]
                             + np.array([inner(x2, x2) for x2 in xs_2])
                             - 2*gram_matrix(inner)(xs_1, xs_2)))
    return rbf_kernel_sigma_inner

if __name__ == '__main__':
    # init data
    X_train, _ = utils.load_data(os.path.join(os.getcwd(),"data/cleaned_tweets_train.tsv"),type="train")
    
    # init params
    beta_low = float(sys.argv[1])
    beta_high = float(sys.argv[2])
    n = int(sys.argv[3])
    betas = np.logspace(beta_low, beta_high, num=n)
    start = time.time()
    for beta in betas:
        if os.path.isfile(f'BoW_G_RBF_gram/BoW_G_RBF_gram_beta_{beta:.1e}.pkl'):
            continue
        BoW_G_RBF_gram = np.zeros((len(X_train), len(X_train)))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(gram_matrix_worker, rbf_kernel_gram, BoW_inner, beta, [x_i], X_train): i for i, x_i in enumerate(X_train)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    BoW_G_RBF_gram[i] = future.result()
                except Exception as exc:
                    print(f'generated an exception: {exc}')
        with open(f"BoW_G_RBF_gram/BoW_G_RBF_gram_beta_{beta:.1e}.pkl", 'wb') as f:
            pickle.dump(BoW_G_RBF_gram, f)
    end = time.time()
    print(f"Time taken: {end-start}")