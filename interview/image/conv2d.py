import numpy as np

def conv2d_full(_input, _kernel):
    """Full convolution with padding"""
    kernel_h, kernel_w = _kernel.shape
    input_h, input_w = _input.shape
    
    # Pad the input
    pad_h = kernel_h - 1
    pad_w = kernel_w - 1
    padded = np.pad(_input, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Perform convolution
    output_h = input_h + kernel_h - 1
    output_w = input_w + kernel_w - 1
    output = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            patch = padded[i:i + kernel_h, j:j + kernel_w]
            output[i, j] = np.sum(patch * _kernel)
    
    return output

def conv2d(_input, _kernel):
    """
    Performs a 2D convolution operation.


    Args:
        _input (np.array): The input 2D image (height, width).
        _kernel (np.array): The 2D convolutional kernel (kernel_height, kernel_width).


    Returns:
        np.array: The convolved output.
    """ 
    kernel_h, kernel_w = _kernel.shape
    input_h, input_w = _input.shape

    range_h, range_w = input_h - kernel_h + 1, input_w - kernel_w + 1
    output = np.zeros((range_h, range_w))

    for i in range(range_h):
        for j in range(range_w):
            input_patch = _input[i: i + kernel_h, j: j + kernel_w]
            element_wise = input_patch *  _kernel
            output[i, j] = np.sum(element_wise)

    return output
    
    
if __name__ == "__main__":
    _input = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])


    _kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])


    output = conv2d(_input, _kernel)
    print(output)