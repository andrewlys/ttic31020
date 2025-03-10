from typing import Tuple, List, Optional
import numpy as np
import utils
from .module import Sequential
from .layer import Conv, batchnorm, ReLU, maxpool, Dropout, flatten, Linear
from .loss import MultiLogisticLoss
from .nn import ERMNeuralNetClassifier

SEED = 0
np.random.seed(SEED)

if __name__ == '__main__':
    train_data = np.load('fmnist_train.npy', allow_pickle=True).item()
    test_data = np.load('fmnist_test.npy', allow_pickle=True).item()

    X = train_data['data']
    y = train_data['labels']
    X_test = test_data['data']

    # Preprocessing X
    X = X[..., np.newaxis] # add channel dimension
    if X.max() > 1: X = X / 255.

    X_test = X_test[..., np.newaxis] # add channel dimension
    if X_test.max() > 1: X_test = X_test / 255.

    # Split into Xfm_train, yfm_train, Xfm_val, yfm_val
    Xfm_train, yfm_train, Xfm_val, yfm_val = utils.create_split(X, y, 0.8)

    # Create model
    model = Sequential([
        Conv(1, 16, 5, 1, 2), 
        batchnorm(16),
        ReLU(),
        maxpool(2, 2), # 14 x 14 x 16
        Dropout(0.1),
        Conv(16, 32, 5, 1, 2),
        batchnorm(32),
        ReLU(),
        maxpool(2, 2), # 7 x 7 x 32
        Dropout(0.1),
        flatten(),
        Linear(7*7*32, 128),
        Dropout(0.5),
        Linear(128, 10)
    ])
    loss = MultiLogisticLoss(k=10)
    cnn_clf = ERMNeuralNetClassifier(model, loss)
    sgd_kwargs = {
        'batch_size': 128,
        'n_epochs': 5,
        'eta': 0.01,
        'verbose': True, # Enable printing INSIDE SGD
        'verbose_epoch_interval': 1,
    }

    # Fit model
    cnn_clf.fit(Xfm_train, yfm_train, **sgd_kwargs)

    # Evaluate model
    print(f"Validation Error: {utils.empirical_err(yfm_val, cnn_clf.predict(Xfm_val))}")
    y_test_preds = cnn_clf.predict(X_test)
    fname = '/content/fmnist_test_pred.csv'
    output = np.vstack((np.arange(y_test_preds.shape[0]), y_test_preds)).T
    np.savetxt(fname, output, fmt="%d", delimiter=',', comments='', header='id,label')