from typing import Tuple, List, Optional
import numpy as np
from .module import Module
from .conv import convolve, convolve_gradient_weight, convolve_gradient_input
from .conv import interweave_with_zeros
SEED = 0
np.random.seed(SEED)

class Linear(Module):
    '''
    Linear transformation on the inputs, z = xW + b.

    Corresponds to all nodes in the preceding layer connected to all nodes in the
    current layer.
    '''
    def __init__(self, in_dim: int, out_dim: int):
        '''
        Args:
            in_dim: Number of input dimensions (number of incoming connections
                in the network).
            out_dim: Number of output dimensions (number of outgoing connections
                in the network).
        '''
        super().__init__()

        # Initialize trainable parameters
        self.weight = np.random.normal(0, np.sqrt(2/in_dim), (in_dim, out_dim))
        self.bias =  np.zeros(out_dim)

        # Initialize gradients w.r.t. trainable parameters
        self._grad_weight = None
        self._grad_bias = None

    def forward(self, x: np.ndarray):
        '''
        Args:
            x: Data features. shape (m, in_dim)
        '''
        assert x.shape[1] == self.weight.shape[0]
        _output = x @ self.weight + self.bias
        if not self.train:
            return _output
        self._input = x
        self._output = _output
        self._check_forward_attrs()
        return _output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        '''
        Computes gradient w.r.t. trainable parameters and input and returns the gradient
        w.r.t. input.

        Important: gradients are accumulated for trainable parameters, i.e. added to the existing
        values.

        Args:
            grad_output: Gradient w.r.t. output dL/dz. shape (m, out_dim)

        Returns:
            grad_input: shape (m, in_dim)
        '''
        assert grad_output.shape[1] == self.weight.shape[1]

        #### TASK 2 CODE
        m = self._input.shape[0]
        self._grad_output = grad_output
        self._grad_weight = (self._input.T @ grad_output)/m #  (grad_output \otimes x) averaged over the samples
        self._grad_bias = np.mean(grad_output, axis = 0)
        self._grad_input = grad_output @ self.weight.T
        #### TASK 2 CODE
        self._check_backward_attrs()

    def _check_backward_attrs(self):
        super()._check_backward_attrs()
        assert self._grad_weight is not None
        assert self._grad_bias is not None

    def get_params(self) -> Optional[np.ndarray]:
        params = np.empty(2, dtype=np.ndarray)
        params[0] = self.weight
        params[1] = self.bias
        return params

    def set_params(self, params: np.ndarray):
        assert len(params) == 2
        self.weight = params[0]
        self.bias = params[1]

    def get_grad_params(self) -> Optional[np.ndarray]:
        grad_params = np.empty(2, dtype=np.ndarray)
        grad_params[0] = self._grad_weight
        grad_params[1] = self._grad_bias
        return grad_params

class ReLU(Module):
    '''
    ReLU activation, not trainable. z = max(x, 0) for each input value x.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray):
        '''
        Args:
            x: Data features. shape (m, in_dim)
        '''
        _output = np.maximum(0., x)
        if not self.train:
            return _output
        self._input = x
        self._output = _output
        self._check_forward_attrs()
        return _output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        '''
        Since there are no trainable parameters, only the gradient w.r.t. input is computed.

        Args:
            grad_output: Gradient w.r.t. output dL/dz. Any shape
        '''
        assert grad_output.shape == self._input.shape

        #### TASK 3 CODE
        self._grad_output = grad_output
        self._grad_input = self._grad_output * (self._input > 0)
        #### TASK 3 CODE
        self._check_backward_attrs()

class Dropout(Module):
    '''
    A dropout layer.
    '''
    def __init__(self, p: float = 0.5):
        '''
        Args:
            p: (default 0.5) Probability of dropping each node (prob. of setting each value to 0).
                If p is 0, then no nodes are dropped, i.e. we get the identity layer.
        '''
        assert 0 <= p <= 1
        super().__init__()
        self.p = p

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.train:
            return x
        self._input = x
        self._output = x
        if self.train and (not np.isclose(self.p, 0)):
            # In training mode and drop probability is positive

            # Create a mask to apply to the input using Bernoulli(1-p) RV
            self.mask = np.random.binomial(1, 1 - self.p, x.shape).astype(float)

            # Scale the mask so that the expected value of prediction during
            # testing is same as x, and not (1-p)x
            self.mask /= 1 - self.p

            self._output *= self.mask
        self._check_forward_attrs()
        return self._output

    def backward(self, grad_output: np.ndarray):
        self._grad_output = grad_output
        self._grad_input = self._grad_output
        if self.train and (not np.isclose(self.p, 0)):
            # In training mode and drop probability is positive, _grad_input is masked
            self._grad_input *= self.mask
        self._check_backward_attrs()

class Conv(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        # hyperparams
        self.in_channels = in_channels
        self.out_channels = out_channels # also number of layers in this conv module
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # params
        self.weight = np.random.normal(0, np.sqrt(2/(in_channels*kernel_size*kernel_size)), 
                                       (out_channels, kernel_size, kernel_size, in_channels)) 
        # we use different kernels for each output channel
        # and we use different kernels for each input channel
        # (out_channels, kernel_size, kernel_size, in_channels) 
        self.bias = np.zeros((1, 1, 1, out_channels)) # broadcast over the dim of the image
        # grad params
        self._grad_weight = None
        self._grad_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the convolutional layer.
        Args:
            x: Data. Shape (m, h, w, in_channels). m is the number of samples, h and w are the height and width of the image, and in_channels is the number of input channels.
            h and w are assumed to be the same.
        Returns:
            np.ndarray: Output of the convolutional layer. Shape (m, h_out, w_out, out_channels). h_out and w_out are the height and width of the output image
            h_out = h_w = (h - kernel_size + 2*padding) // stride + 1
        '''
        assert x.ndim == 4 # x.shape = (m, h, w, in_channels)
        assert x.shape[-1] == self.in_channels # check if the number of input channels is correct
        assert x.shape[1] == x.shape[2] # check if the input is square
        assert self.weight.shape[3] == self.in_channels # check if the number of input channels in the weight is correct
        _output = convolve(x, self.weight, self.stride, self.padding) + self.bias
        if not self.train:
            return _output
        self._input = x
        self._output = _output
        self._check_forward_attrs()
        return _output
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        '''
        Computers gradient of loss w.r.t. input and parameters.
        Args:
            grad_output: Gradient of loss w.r.t. output of the module. Shape (m, h_out, w_out, out_channels)
        dL/do = dL/do
        dL/dbias = dL/do
        dL/dx = dilate(pad(dL/do, stride - 1), stride - 1) * rot180(kernel)
        dL/dkernel = dilate(dL/do, stride - 1) * x
        '''
        # This blog helped me understand the backpropagation of a convolutional layer
        # https://grzegorzgwardys.wordpress.com/2016/04/22/8/
        # these blog posts helped me get the dilation and padding right
        # https://hideyukiinada.github.io/cnn_backprop_strides2.html
        # https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3
        # set params
        self._grad_output = grad_output
        self._grad_bias = np.mean(grad_output, axis = 0) # DON'T FORGET TO TAKE THE MEAN OVER THE SAMPLES
        self._grad_input = convolve_gradient_input(self._input, self.weight, grad_output, self.stride, self.padding)
        self._grad_weight = convolve_gradient_weight(self._input, self.weight, grad_output, self.stride, self.padding)/self._input.shape[0] # DON'T FORGET TO TAKE THE MEAN OVER THE SAMPLES
        # sanity check
        self._check_backward_attrs()

    def _check_backward_attrs(self):
        super()._check_backward_attrs()
        assert self._grad_weight is not None
        assert self._grad_bias is not None

    def get_params(self) -> Optional[np.ndarray]:
        params = np.empty(2, dtype=np.ndarray)
        params[0] = self.weight
        params[1] = self.bias
        return params

    def set_params(self, params):
        assert len(params) == 2
        self.weight = params[0]
        self.bias = params[1]

    def get_grad_params(self) -> Optional[np.ndarray]:
        grad_params = np.empty(2, dtype=np.ndarray)
        grad_params[0] = self._grad_weight
        grad_params[1] = self._grad_bias
        return grad_params

class maxpool(Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # param to account for intermediary step in forward pass
        # inspired by this blog post
        # https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3
        self._dil_input = None # shape (m, h_out, w_out, kernel_size, kernel_size, c)
    def forward(self, x: np.ndarray) -> np.ndarray:
        ''''
        Forward pass of the pooling layer.'
        '''
        assert x.ndim == 4
        # init stuff
        m, h, w, c = x.shape
        m_stride, h_stride, w_stride, c_stride = x.strides
        h_out = (h - self.kernel_size) // self.stride + 1
        w_out = (w - self.kernel_size) // self.stride + 1
        # stuff done here is done with EXTREME CARE
        # EVIL vectorized code
        dil_shape = (m, h_out, w_out, self.kernel_size, self.kernel_size, c)
        dil_strides = (m_stride, h_stride * self.stride, w_stride * self.stride, h_stride, w_stride, c_stride)
        # cache stuff
        _dil_input= np.lib.stride_tricks.as_strided(x, dil_shape, dil_strides)
        _output = np.max(_dil_input, axis = (3, 4))
        if not self.train:
            return _output
        self._input = x
        self._dil_input = _dil_input
        self._output = np.max(_dil_input, axis = (3, 4))
        self._check_forward_attrs()
        return _output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        '''
        Computes gradient of loss w.r.t. input.
        Args:
            grad_output: Gradient of loss w.r.t. output of the module. Shape (m, h_out, w_out, c)
        '''
        self._grad_output = grad_output
        self._grad_input = np.zeros_like(self._input)
        # mask
        m, h_out, w_out, k_h, k_w, c = self._dil_input.shape
        flat_dil = self._dil_input.reshape(m, h_out, w_out, k_h * k_w, c)
        flat_idx = np.argmax(flat_dil, axis = 3, keepdims=True)
        mask = np.eye(k_h * k_w)[flat_idx].reshape(m, h_out, w_out, k_h, k_w, c)
        dil_grad_out = grad_output[:, :, :, np.newaxis, np.newaxis, :] * mask
        # more evil vectorized code
        m, h, w, c = self._grad_input.shape
        m_str, h_str, w_str, c_str = self._grad_input.strides
        h_out = (h - k_h) // self.stride + 1
        w_out = (w - k_w) // self.stride + 1
        dil_shape = (m, h_out, w_out, k_h, k_w, c)
        dil_strides = (m_str, h_str * self.stride, w_str * self.stride, h_str, w_str, c_str)
        dil_grad_in = np.lib.stride_tricks.as_strided(self._grad_input, dil_shape, dil_strides)
        dil_grad_in += dil_grad_out # add the contributions of the gradient
        # result will automatically be updated in the _grad_input attribute, since dil_grad_in is a view of _grad_input
        self._check_backward_attrs()

class batchnorm(Module):
    def __init__(self, channels, eps = 1e-5):
        # https://arxiv.org/pdf/1502.03167
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.mean = np.zeros((1,1,1,channels))
        self.var = np.ones((1,1,1,channels))
        # learnable params
        self.weight = np.ones((1,1,1,channels))
        self.bias = np.zeros((1,1,1,channels))
        # grad params
        self._grad_weight = None
        self._grad_bias = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the batchnorm layer.
        Args:
            x: Data. Shape (m, h, w, channels). m is the number of samples, h and w are the height and width of the image, and channels is the number of channels.
            h and w are assumed to be the same.
        '''
        assert x.ndim == 4
        mean = np.mean(x, axis = (0, 1, 2), keepdims = True)
        var = np.var(x, axis = (0, 1, 2), keepdims = True)
        _output = (x - mean) / np.sqrt(var + self.eps) * self.weight + self.bias
        if not self.train:
            return _output
        self._input = x
        self.mean = mean
        self.var = var
        self._output = _output
        self._check_forward_attrs()
        return _output
    
    def backward(self, grad_output):
        '''
        Computes gradient of loss w.r.t. input and parameters.
        Args:
            grad_output: Gradient of loss w.r.t. output of the module. Shape (m, h, w, channels)
        '''
        # inspired by this blog post
        # https://cthorey.github.io./blog/2016/backprop_conv/
        self._grad_output = grad_output
        self._grad_bias = np.sum(grad_output, axis = (0, 1, 2), keepdims = True)
        self._grad_weight = np.sum(self._output * self._grad_output, axis = (0, 1, 2), keepdims = True)
        term1 = self.weight * 1. / np.sqrt(self.var + self.eps)
        term2 = np.mean(self._grad_output, axis = (0, 1, 2), keepdims = True)
        term3 = np.mean(self._grad_output * (self._input - self.mean), axis = (0, 1, 2), keepdims = True)
        self._grad_input = term1 * (self._grad_output - term2) - term1 * term3 * (self._input - self.mean) / (self.var + self.eps)
        self._check_backward_attrs()
    
    def _check_backward_attrs(self):
        super()._check_backward_attrs()
        assert self._grad_weight is not None
        assert self._grad_bias is not None

    def get_params(self) -> Optional[np.ndarray]:
        params = np.empty(2, dtype=np.ndarray)
        params[0] = self.weight
        params[1] = self.bias
        return params

    def set_params(self, params):
        assert len(params) == 2
        self.weight = params[0]
        self.bias = params[1]

    def get_grad_params(self) -> Optional[np.ndarray]:
        grad_params = np.empty(2, dtype=np.ndarray)
        grad_params[0] = self._grad_weight
        grad_params[1] = self._grad_bias
        return grad_params

class flatten(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass of the flatten layer.
        Args:
            x: Data. Shape (m, h, w, channels). m is the number of samples, h and w are the height and width of the image, and channels is the number of channels.
            h and w are assumed to be the same.
        '''
        assert x.ndim == 4
        _output = x.reshape(x.shape[0], -1)
        if not self.train:
            return _output
        self._input = x
        self._output = x.reshape(x.shape[0], -1)
        self._check_forward_attrs()
        return _output
    
    def backward(self, grad_output):
        '''
        Computes gradient of loss w.r.t. input.
        Args:
            grad_output: Gradient of loss w.r.t. output of the module. Shape (m, h, w, channels)
        '''
        self._grad_output = grad_output
        self._grad_input = grad_output.reshape(self._input.shape)
        self._check_backward_attrs()
