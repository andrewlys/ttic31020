import numpy as np
import utils
from .module import Loss
SEED = 0
np.random.seed(SEED)

class MultiLogisticLoss(Loss):
    def __init__(self, k: int):
        super(MultiLogisticLoss, self).__init__(k)

    def forward(self, r: np.ndarray, y: np.ndarray):
        '''
        Computes the multiclass logistic loss, using the softmax operation to
        convert responses r to normalized probabilities.
        '''
        assert r.shape[0] == y.shape[0]
        assert r.shape[1] == self.k

        self._input = r
        self._input_target = y

        stable_r = self._input - np.max(self._input, axis=1)[:, np.newaxis]
        nll = np.log(np.sum(np.exp(stable_r), axis=1)) - \
            np.take_along_axis(stable_r, self._input_target[:, np.newaxis], axis=1).flatten()
        self._output = np.mean(nll)
        self._check_forward_attrs()

    def backward(self):
        #### TASK 4 CODE
        self._grad_input = utils.softmax(self._input) - np.eye(self.k)[self._input_target]
        self._grad_input /= self._input.shape[0]
        #### TASK 4 CODE
        self._check_backward_attrs()