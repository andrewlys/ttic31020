import abc
from typing import Tuple, List, Optional
import numpy as np
SEED = 0
np.random.seed(SEED)

class Module(abc.ABC):
    '''
    A module defines a sub-graph G' = (V, E) in the DAG G that represents a
    neural network. Therefore G' is a subset of G. An instantiation of a module
    thus represents a realization of a subgraph, comprising of a set of nodes and
    their connections.
    '''
    def __init__(self):
        # Train mode or eval mode
        self.train = True

        # Forward pass cache
        self._input = None
        self._output = None

        # Backward pass cache
        self._grad_output = None
        self._grad_input = None

    @abc.abstractmethod
    def forward(self, x: np.ndarray):
        '''
        Computes the forward pass z = f(x) where f is the function represented by
        the module, x is the input, and z is the output of the forward pass.

        Assigns to attributes self._input and self._output.

        Args:
            x: Data features. shape (m, in_dim) where m is the number of data
                points and in_dim is the number of input features.
        '''
        pass

    def _check_forward_attrs(self):
        '''Sanity check after a forward pass.'''
        assert self._input is not None
        assert self._output is not None

    @abc.abstractmethod
    def backward(self, grad_output: np.ndarray):
        '''
        Computes the gradient of loss w.r.t. cached input and trainable parameters.

        Assigns to attributes self._grad_output and self._grad_input.
        The gradients w.r.t. trainable parameters must also be cached so that
        they can be returned by self.get_grad_params().

        Args:
            grad_output: Gradient of loss w.r.t. output z of the module, dL/dz.
                shape (m, out_dim) where m is the number of data points and
                out_dim is the number of output features.
        '''
        pass

    def _check_backward_attrs(self):
        '''Sanity check after a backward pass.'''
        assert self._grad_output is not None
        assert self._grad_input is not None

    def get_params(self) -> Optional[np.ndarray]:
        '''
        Returns the trainable parameters of the module. If there are no trainable
        parameters, returns None.

        Returns:
            arr: (jagged) array of trainable parameters, where the entries are
                differently-sized numpy arrays themselves.
        '''
        return None

    def set_params(self, params: np.ndarray):
        '''
        Sets the trainable parameters to params. If there are no trainable parameters
        to set, raises a RuntimeError.

        Args:
            params: (jagged) array of trainable parameters, in the same order
                as obtained from self.get_params(). The identity operation is satisfied:
                ```
                x = self.get_params()
                self.set_params(x)
                x == self.get_params()
                ```
        '''
        raise RuntimeError('No trainable parameters to set!')

    def get_grad_params(self) -> Optional[np.ndarray]:
        '''
        Returns the gradients of the loss w.r.t. trainable parameters of the module.
        If there are no trainable parameters, returns None.

        Returns:
            arr: (jagged) array of gradients of trainable parameters,
                where the entries are differently-sized numpy arrays themselves.
        '''
        return None

    def train_mode(self):
        '''
        Switches on the training mode. Useful e.g. in Dropout, where the nodes must be
        dropped only during training, not evaluation.
        '''
        self.train = True

    def eval_mode(self):
        '''
        Switches on the evaluation mode. Useful e.g. in Dropout, where the nodes must be
        dropped only during training, not evaluation.
        '''
        self.train = False

class Sequential(Module):
    '''
    A sequence of modules, representing a DAG path.
    '''
    def __init__(self, layers: List[Module]):
        '''
        Args:
            layers: A list of modules to initialize the sequential network.
        '''
        super(Sequential, self).__init__()
        self.layers = layers

    def add(self, layer: Module):
        '''
        Adds the layer at the end of the sequential network.
        '''
        self.layers.append(layer)

    def forward(self, x: np.ndarray):
        '''
        Computes a forward pass sequentially on the network layers.
        '''
        self._input = x
        n_layers = len(self.layers)
        for i in range(n_layers):
            _output_prev = self._input if i == 0 else self.layers[i-1]._output
            self.layers[i].forward(_output_prev)
        self._output = self._input if n_layers == 0 else self.layers[-1]._output

        self._check_forward_attrs()

    def backward(self, grad_output: np.ndarray):
        '''
        Backpropagates the gradient w.r.t. output of the sequential network,
        computing gradients w.r.t. input and trainable parameters of the network.
        '''
        #### TASK 1 CODE
        self._grad_output = grad_output
        n_layers = len(self.layers)
        for i in range(n_layers-1, -1, -1):
            _grad_output_next = self._grad_output if i == n_layers-1 else self.layers[i+1]._grad_input
            self.layers[i].backward(_grad_output_next)
        self._grad_input = self.grad_output if n_layers == 0 else self.layers[0]._grad_input
        #### TASK 1 CODE
        self._check_backward_attrs()

    def get_params(self) -> Optional[np.ndarray]:
        params = []
        for layer in self.layers:
            p = layer.get_params()
            if p is not None:
                params.append(p)

        # Wrap parameters in an array. Just np.array(params) won't work due to broadcasting
        # conflicts: https://stackoverflow.com/a/49119983. So initialize array and then fill.
        arr = np.empty(len(params), dtype=np.ndarray)
        arr[:] = params
        return arr

    def set_params(self, params: np.ndarray):
        # Since params has trainable parameters listed in the same order as
        # get_params() would have returned, follow the same iteration, and call
        # layer.set_params() on params[i] where i is the ith layer with any trainable
        # parameters
        i = 0
        for layer in self.layers:
            p = layer.get_params()
            if p is not None:
                layer.set_params(params[i])
                i += 1

    def get_grad_params(self) -> Optional[np.ndarray]:
        grad_params = []
        for layer in self.layers:
            g = layer.get_grad_params()
            if g is not None:
                grad_params.append(g)
        arr = np.empty(len(grad_params), dtype=np.ndarray)
        arr[:] = grad_params
        return arr

    def train_mode(self):
        # Switch on training in all layers
        super().train_mode()
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self):
        # Switch on eval in all layers
        super().eval_mode()
        for layer in self.layers:
            layer.eval_mode()

class Loss(abc.ABC):
    '''Defines a loss function.'''
    def __init__(self, k: int):
        '''
        Args:
            k: Number of labels.
        '''
        self.k = k
        self._input = None
        self._input_target = None
        self._output = None
        self._grad_input = None

    @abc.abstractmethod
    def forward(self, r: np.ndarray, y: np.ndarray):
        '''
        Computes the loss value using responses r and true labels y.

        Sets the attributes self._input, self._input_target, and self._output.

        Args:
            r: Responses of a classifier. shape (m, k) where m is the number of data
                points.
            y: True labels. shape (m). For all i, 0 <= y_i < k
        '''
        pass

    def _check_forward_attrs(self):
        assert self._input is not None
        assert self._input_target is not None
        assert self._output is not None

    @abc.abstractmethod
    def backward(self):
        '''
        Computes the gradient of the loss value w.r.t. cached responses.

        Sets the attribute self._grad_input.
        '''
        pass

    def _check_backward_attrs(self):
        assert self._grad_input is not None