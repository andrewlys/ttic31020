import numpy as np
SEED = 0
np.random.seed(SEED)

def pad(image: np.ndarray, padding: int) -> np.ndarray:
    '''
    Pad image with zeros
    Args:
        image: 4D array of shape (m, h, w, c)
        padding: int. Number of zeros to pad with
    Returns:
        4D array of shape (m, h + 2*padding, w + 2*padding, c)
    '''
    m, h, w, c = image.shape
    res = np.zeros((m, h + 2*padding, w + 2*padding, c))
    res[:, padding:h+padding, padding:w+padding, :] = image
    return res

def convolve(image: np.ndarray, kernel: np.ndarray, stride: int, padding: int):
    '''
    Convolve batch of layers of 2D image/array with kernel. 
    Args:
        image: 4D array of shape (m, h, w, in_channels)
        kernel: 4D array of shape (out_channels, k_h, k_w, in_channels)
        as we go backwards, kernel shape is (in_channels, k_h, k_w, out_channels)
        k_h < h, k_w < w
        stride: int. How much kernel slides over by
        padding: int. How much to pad the image with zeros
    returns:
        4D array of shape (m, h_out, w_out, out_channels)
    '''
    # get dimensions
    m, h, w, c_i = image.shape
    o_c, k_h, k_w, c_k = kernel.shape
    assert c_i == c_k, f"in_channels of the image and kernel must be the same, got {c_i} and {c_k}"
    # get output dimensions
    h_out = (h - k_h + 2*padding) // stride + 1
    w_out = (w - k_w + 2*padding) // stride + 1
    # pad image
    pad_image = pad(image, padding)
    # evil vectorized code
    m_str, h_str, w_str, c_str = pad_image.strides
    dil_shape = (m, h_out, w_out, k_h, k_w, c_i)
    dil_strides = (m_str, h_str * stride, w_str * stride, h_str, w_str, c_str)
    dil_image = np.lib.stride_tricks.as_strided(pad_image, dil_shape, dil_strides)
    return np.einsum('mhwkdc,okdc->mhwo', dil_image, kernel, optimize=True)

def convolve_gradient_weight(image: np.ndarray, kernel: np.ndarray, grad: np.ndarray, stride: int, padding: int) -> np.ndarray:
    '''
    Compute the gradient of the loss w.r.t. the weights of the convolutional layer.
    Args:
        image: 4D array of shape (m, h, w, in_channels)
        kernel: 4D array of shape (out_channels, k_h, k_w, in_channels)
        grad: 4D array of shape (m, h_out, w_out, out_channels)
        stride: int. How much kernel slides over by
        padding: int. How much to pad the image with zeros
    returns:
        4D array of shape (out_channels, k_h, k_w, in_channels)
    '''
    m, h, w, c_i = image.shape
    m, h_out, w_out, c_o = grad.shape
    c_o, k_h, k_w, c_i = kernel.shape
    # pad image
    image = pad(image, padding)
    # get output dimensions
    h_out = (h - k_h + 2*padding) // stride + 1
    w_out = (w - k_w + 2*padding) // stride + 1
    # stride
    m_str, h_str, w_str, c_str = image.strides
    dil_shape = (m, h_out, w_out, k_h, k_w, c_i)
    dil_strides = (m_str, h_str * stride, w_str * stride, h_str, w_str, c_str)
    image = np.lib.stride_tricks.as_strided(image, dil_shape, dil_strides)
    return np.einsum('mhwijc,mhwo->oijc', image, grad)

def convolve_gradient_input(image: np.ndarray, kernel: np.ndarray, grad: np.ndarray, stride: int, padding: int) -> np.ndarray:
    '''
    Compute the gradient of the loss w.r.t. the input of the convolutional layer.
    Args:
        image: 4D array of shape (m, h, w, in_channels)
        kernel: 4D array of shape (out_channels, k_h, k_w, in_channels)
        grad: 4D array of shape (m, h_out, w_out, out_channels)
        stride: int. How much kernel slides over by
        padding: int. How much to pad the image with zeros
    returns:
        4D array of shape (m, h, w, in_channels)
    '''
    m, h, w, c_i = image.shape
    m, h_out, w_out, c_o = grad.shape
    c_o, k_h, k_w, c_i = kernel.shape
    dil_grad_input = np.zeros((m, h + 2*padding, w + 2*padding, c_i))
    m_str, h_str, w_str, c_str = dil_grad_input.strides
    dil_shape = (m, h_out, w_out, k_h, k_w, c_i)
    dil_strides = (m_str, h_str * stride, w_str * stride, h_str, w_str, c_str)
    dil_grad_input = np.lib.stride_tricks.as_strided(dil_grad_input, dil_shape, dil_strides)
    dil_grad_input += np.einsum('oijc, mhwo->mhwijc', kernel, grad)
    return np.lib.stride_tricks.as_strided(dil_grad_input, (m, h, w, c_i), (m_str, h_str, w_str, c_str))


def interweave_with_zeros(arr: np.ndarray, n: int) -> np.ndarray:
    '''
    Add a int rows and cols of zeros after each element in the array.
    Args:
        np.ndarray: 4D array
        shape (m, h, w, c)
        n: number of zeros to add after each element
    Returns:
        np.ndarray: 2D array with zeros added
        shape (m, h*(n+1) - 1, w*(n+1) - 1, c)
    '''
    m, h, w, c= arr.shape
    if n == 0:
        return arr
    new_h = h + h*n - 1
    new_w = w + w*n - 1
    new_arr = np.zeros((m, new_h, new_w, c))
    new_arr[:, 0::n+1,0::n+1, :] = arr
    return new_arr