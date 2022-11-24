import numpy as np

def conv2d(img, kernel):
    """
        Ultra slow conv-2d
    """
    n, h, w, c = img.shape
    k, k, c_i, c_o = kernel.shape
    res = np.zeros((n, h-k+1, w-k+1, c_o))
    for n_ in range(n):
        for c_out in range(c_o):
            for y in range(h-(k-1)):
                for x in range(h-(k-1)):
                    snip = img[n_, y:y+k, x:x+k,:]
                    res[n_, y,x,c_out] = np.sum(snip * kernel[:,:,:,c_out])
    return res

def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling, 
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=np.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

class Conv2D:
    def __init__(self, kernel, bias):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, img):
        kernel = self.kernel
        n, h, w, c = img.shape
        k, k, c_i, c_o = kernel.shape
        res = np.zeros((n, h-k+1, w-k+1, c_o))
        for n_ in range(n):
            for c_out in range(c_o):
                for y in range(h-(k-1)):
                    for x in range(h-(k-1)):
                        snip = img[n_, y:y+k, x:x+k,:]
                        res[n_, y,x,c_out] = np.sum(snip * kernel[:,:,:,c_out])
        return res + self.bias

class MaxPool2D:
    def __init__(self, stride):
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
    
    def __call__(self, img):
        n, h, w, c = img.shape
        k = self.stride
        res = np.empty(n, h//k , w//k, c)
        for n_ in range(n):
            res[n_] = pooling(img[n_], (k, k))

class Flatten:
    def __call__(self, img):
        n, h, w, c = img.shape
        return img.reshape(n, -1)

class Dense:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, img):
        return (img @ self.weight) + self.bias

class Softmax:
    def __init__(self, axis = -1):
        self.axis = axis

    def __call__(self, x):
        axis = self.axis

        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

class ReLU:
    def __call__(self, x):
        x[x<0] = 0
        return x

class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x