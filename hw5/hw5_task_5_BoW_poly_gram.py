import os
import numpy as np
import utils
import concurrent.futures
import pickle 
import time 
import sys

def inner_matrix_worker(inner, xs_1, xs_2):
    return np.array([[inner(x1, x2) for x2 in xs_2] for x1 in xs_1])
def BoW_inner(s1,s2):
    "returns inner product between bag-of-word feature vectors of the two input strings"
    from collections import Counter
    d1 = Counter(s1.split())
    return sum(d1[w] for w in s2.split())
def poly_ker(G, alpha, deg):
    if os.path.isfile(f'BoW_poly_gram/bow_gram_poly_deg_{deg}_alpha_{alpha:.0e}.pkl'):
        return
    bow_poly_gram = (G + alpha)**deg
    bow_poly_gram = (bow_poly_gram - np.mean(bow_poly_gram))/np.std(bow_poly_gram)
    with open(f"BoW_poly_gram/bow_gram_poly_deg_{deg}_alpha_{alpha:.0e}.pkl", 'wb') as f:
                pickle.dump(bow_poly_gram, f)
    return

if __name__ == '__main__':
    # init data
    X_train, _ = utils.load_data(os.path.join(os.getcwd(),"data/cleaned_tweets_train.tsv"),type="train")
    # receive params
    alpha_low = float(sys.argv[1])
    alpha_high = float(sys.argv[2])
    n_alpha = int(sys.argv[3])
    deg_low = int(sys.argv[4])
    deg_high = int(sys.argv[5])
    block_size = int(sys.argv[6])
    # init params
    alphas = np.logspace(alpha_low, alpha_high, n_alpha)
    degs = np.arange(deg_low, deg_high + 1)
    bow_gram = np.zeros((len(X_train), len(X_train)))
    # Compute inner gram
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(inner_matrix_worker, BoW_inner, X_train[block_size*i:block_size*(i + 1)], X_train): i for i in range(int(len(X_train)/block_size)) }
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                bow_gram[block_size*i:block_size*(i+1)] = future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')
    # compute poly_ker and save
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(poly_ker, bow_gram, alpha, deg) for deg in degs for alpha in alphas}
    end = time.time()
    print(f"Time taken: {end-start}")