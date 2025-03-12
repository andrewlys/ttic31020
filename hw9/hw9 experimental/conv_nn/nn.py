from typing import Tuple, List, Optional
import linclass
import numpy as np
from . import utils
from .module import Module
from .module import Loss
from functools import partial

SEED = 0
np.random.seed(SEED)

def train_grad(X, y, model, loss, params: np.ndarray, batch: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Returns the gradient of the training objective w.r.t. parameters,
    calculated on a batch of training samples.

    Args:
        params: Trainable parameters, in the same format as self.model.get_params().
        batch: (default None) Indices of samples to calculate objective on. If None,
            calculate objective on all samples.
    '''
    if batch is None:
        # All data is in a batch
        batch = slice(None)

    model.set_params(params)
    #### TASK 5 CODE
    # Forward pass
    model.forward(X[batch])
    loss.forward(model._output, y[batch])

    # Backward pass
    loss.backward()
    model.backward(loss._grad_input)

    grad_params = model.get_grad_params() #need to take mean over the samples

    #### TASK 5 CODE
    return grad_params

class ERMNeuralNetClassifier(linclass.Classifier):
    '''
    Neural network trained by minimizing the empirical risk with SGD,
    w.r.t. some loss function.
    '''
    def __init__(self, model: Module, loss: Loss, **kwargs):
        '''
        Args:
            model: A neural network object with initialized parameters.
            loss: A loss function.
        '''
        super().__init__(**kwargs)
        self.model = model
        self.params0 = self.model.get_params()
        self.params = None
        self.loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Returns predicted labels for data.

        Args:
            X: Data features. shape (m, d_in) where d_in is the number of
                input features of self.model.

        Returns:
            shape (m)
        '''
        assert self.params is not None, "Classifier hasn't been fit!"

        # Switch to evaluation mode to disable dropout
        self.model.eval_mode()

        #### TASK 5 CODE
        self.model.forward(X)
        return self.model._output.argmax(axis = 1) # return the label with the highest probability
        #### TASK 5 CODE

    def fit(self, X: np.ndarray, y: np.ndarray, **sgd_kwargs):
        '''
        Fits the classifier on dataset.

        Args:
            X: Data features. shape (m, d_in) where d_in is the number of input
                features of self.model.
            y: Data labels, 0 <= y_i < k. shape (m)
        '''
        assert X.shape[0] == y.shape[0]

        m = X.shape[0]
        # Define training objective
        def train_obj(params: np.ndarray, batch: Optional[np.ndarray] = None) -> float:
            '''
            Calculates the training objective with parameters on a batch of training samples.

            Args:
                params: Trainable parameters, in the same format as self.model.get_params().
                batch: (default None) Indices of samples to calculate objective on. If None,
                    calculate objective on all samples.
            '''
            if batch is None:
                # All data is in a batch
                batch = slice(None)

            self.model.set_params(params)

            # Forward pass
            self.model.forward(X[batch])
            self.loss.forward(self.model._output, y[batch])

            loss_val = self.loss._output
            return loss_val
        
        train_grad_bound = partial(train_grad, X, y, self.model, self.loss)

        # Define training gradient

        self.sgd_loggers = [
            utils.SGDLogger('train_obj', train_obj, can_display=True, per_epoch=False),
        ] + sgd_kwargs.pop('loggers', [])

        # Switch to training mode to enable dropout, if present in the model
        self.model.train_mode()

        # Optimize using SGD
        self.params = utils.SGD(
            self.params0,
            train_grad_bound,
            m,
            loggers=self.sgd_loggers,
            **sgd_kwargs
        )