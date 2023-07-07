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

def pooling(mat,ksize,method='max',pad=False, lrp = False):
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

        # Calculates which index max pooling got the value from
        # Only works with non overlapping kernels
        if lrp == True:
            c = mat.shape[2]
            lrp_source = np.zeros_like(mat, dtype = bool) # Boolean mask

            x_argmax = np.nanargmax(mat_pad.reshape(new_shape),axis=(3))

            # Arrange index selectors with offsets
            y_vals = np.arange(kx)
            y_vals = np.tile(np.arange(kx), kx*c)
            y_offset = np.tile(np.tile(np.repeat(np.arange(nx), kx) * kx, ny), c)
            x_offset = np.tile(np.repeat(np.repeat(np.arange(nx), kx) * kx, ny), c)
            c_vals = np.repeat(np.arange(c), kx*ky)

            # Shuffle index so index values go from column wise kernels, then rows, then channels
            x_vals = np.moveaxis(x_argmax, [0,1,2,3], (2, 3, 1, 0)).flatten()
            
            x_ind = x_vals + x_offset
            y_ind = y_vals + y_offset
            
            # Get max in each kernel
            best_in_kernel = mat[y_ind, x_ind, c_vals].reshape(4, -1, order = 'F').argmax(0)

            # Retrieve max index for each kernel pass
            final_x = x_ind.reshape(4, -1, order = 'F')[best_in_kernel, np.arange(nx*ny*c)]
            final_y = y_ind.reshape(4, -1, order = 'F')[best_in_kernel, np.arange(nx*ny*c)]

            # Set result
            lrp_source[final_y, final_x, np.repeat(np.arange(c), nx*ny)] = True
            
            return result, np.nanargmax(mat_pad.reshape(new_shape),axis=(3)), lrp_source
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result, np.nanargmax(mat_pad.reshape(new_shape),axis=(3))

class Conv2D:
    def __init__(self, kernel, bias, input_shape, lrp_on = False):
        self.kernel = kernel
        self.bias = bias
        self.lrp_on = lrp_on

        k1, k2, c_in, c_out = kernel.shape
        n, h, w, c = input_shape

        template = np.arange(h*w*c).reshape(1,h,w,c) # Template for 1D indexing of input image
        snips = []
        for y in range(h-(k1-1)):
            for x in range(h-(k2-1)):
                snip = template[0, y:y+k1, x:x+k2,:]
                snips.append(snip)
        self.img_inds = np.asarray(snips).reshape(-1, k1 * k2* c_in)
        self.kernel = kernel.reshape(k1*k2*c_in, c_out)
        self.output_shape = (-1, h-k1+1, w-k2+1, c_out)

    def __call__(self, img):
        img = img.reshape(-1) # Flatten for indexing
        img = img[self.img_inds]
        img = img @ self.kernel
        img = img.reshape(*self.output_shape)
        return img + self.bias

    def lrp(self, a):
        return a * self.last_call

    def clear(self):
        self.last_call = None

class MaxPool2D:
    def __init__(self, stride, lrp_on = False):
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.lrp_on = lrp_on
    
    def __call__(self, img):
        n, h, w, c = img.shape
        h_k, w_k = self.stride
        res = np.empty((n, h//h_k , w//w_k, c))
        for n_ in range(n):
            res[n_] = pooling(img[n_], (h_k, w_k))
        return res

    def lrp(self, a):
        return a * self.last_call

    def clear(self):
        self.last_call = None

class Flatten:
    def __init__(self, lrp_on = False):
        self.lrp_on = lrp_on

    def __call__(self, img):
        n, h, w, c = img.shape
        return img.reshape(n, -1)

class Dense:
    def __init__(self, weight, bias, lrp_on = False):
        self.weight = weight
        self.bias = bias
        self.lrp_on = lrp_on

    def __call__(self, img):
        return (img @ self.weight) + self.bias

    def lrp(self, a):
        return a * self.last_call

    def clear(self):
        """
            No need for implementation for linear unit
        """
        pass

class Softmax:
    def __init__(self, axis = -1, lrp_on = False):
        self.axis = axis
        self.lrp_on = lrp_on
        self.last_call = None

    def __call__(self, x):
        axis = self.axis

        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        y = y / y.sum(axis=axis, keepdims=True)
        self.last_call = y

        return y
    
    def lrp(self, a):
        return a * self.last_call

    def clear(self):
        self.last_call = None

class ReLU:
    def __init__(self, lrp_on = False):
        self.last_call = None
        self.lrp_on = lrp_on

    def __call__(self, x):
        x[x<0] = 0
        return x

    def lrp(self, a):
        return a * self.last_call

    def clear(self):
        self.last_call = None

class Model:
    def __init__(self, layers:list):
        self.layers = layers
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def clear(self):
        for l in self.layers:
            l.clear()

    def lrp_on(self):
        for l in self.layers:
            l.lrp_on = True

    def lrp_off(self):
        for l in self.layers:
            l.lrp_on = False
    
    def lrp(self):
        a = self.layers[-1].a

        for l in self.layers[:-1:-1]:
            a = l.lrp(a)
        
        return a