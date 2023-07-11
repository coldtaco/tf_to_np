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
        if lrp:
            c = mat.shape[2]
            lrp_source = np.zeros_like(mat, dtype = bool) # Boolean mask

            # (ny, ky, nx, c)
            x_argmax = np.nanargmax(mat_pad.reshape(new_shape),axis=(3))

            # Arrange index selectors with offsets
            y_vals = np.arange(kx)
            y_vals = np.tile(np.arange(kx), nx*ny*c)
            y_offset = np.tile(np.tile(np.repeat(np.arange(nx), kx) * kx, ny), c)
            x_offset = np.tile(np.repeat(np.repeat(np.arange(nx), kx) * kx, ny), c)
            c_vals = np.repeat(np.arange(c), kx*nx*ny)

            # Shuffle index so index values go from column wise kernels, then rows, then channels
            x_vals = np.moveaxis(x_argmax, [0,1,2,3], (2, 3, 1, 0)).flatten()
            
            x_ind = x_vals + x_offset
            y_ind = y_vals + y_offset
            
            # Get max in each kernel
            best_in_kernel = mat[y_ind, x_ind, c_vals].reshape(ky, -1, order = 'F').argmax(0)

            # Retrieve max index for each kernel pass
            final_x = x_ind.reshape(ky, -1, order = 'F')[best_in_kernel, np.arange(nx*ny*c)]
            final_y = y_ind.reshape(ky, -1, order = 'F')[best_in_kernel, np.arange(nx*ny*c)]

            # Set result
            lrp_source[final_y, final_x, np.repeat(np.arange(c), nx*ny)] = True
            
            return result, lrp_source
    else:
        result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

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

    def __call__(self, img:np.ndarray, lrp = False):
        img = img.reshape(-1) # Flatten for indexing
        img = img[self.img_inds]
        # (n * h-k1+1 * w-k2+1, k1*k2*c_in, c_out)
        img = img @ self.kernel


        img = img.reshape(*self.output_shape)
        img = img + self.bias

        
        return img

    def lrp_backward_old(self, relevance:np.ndarray):
        n = relevance.shape[0]
        # TODO change so kernel size is not hard coded
        input_shape = (n, self.output_shape[1] + 4, self.output_shape[2] + 4, self.kernel.shape[0]//25)
        output_shape = [n] + list(self.output_shape)[1:]
        lrp_score = np.zeros(input_shape)

        n_pixels = 1
        for x in input_shape[1:]:
            n_pixels *= x

        result = np.zeros((n, n_pixels))
        print(input_shape, result.shape, relevance.shape)

        relevance = relevance.reshape(n, -1)

        counter = 0
        # Each image in batch
        for n_i in range(n):
            # Each pixel in output image
            for p in range(output_shape[1] * output_shape[2]):
                # Each channel in output image
                for c_out in range(output_shape[-1]):
                    a = self.last_call[counter] # Image already sliced for per pixel output
                    w = self.kernel[:,c_out] # Kernel stack for specific pixel in output
                    aw = a * w
                    aw /= aw.sum()

                    source_ind = self.img_inds[p] # Which pixels used in source image

                    rel = relevance[n_i, p * output_shape[-1] + c_out] # Get relevance of specific pixel

                    result[n_i, source_ind] += aw * rel

        return result.reshape(input_shape)

    def lrp_backward(self, relevance:np.ndarray):
        n = relevance.shape[0]
        # TODO change so kernel size is not hard coded
        input_shape = (n, self.output_shape[1] + 4, self.output_shape[2] + 4, self.kernel.shape[0]//25)

        a = self.last_call
        z = self(a) + 1e-9# Denominator
        s = relevance/z

        # relevance -> output shape
        # weights -> weights per output pixel
        print(s.shape)
        print(self.kernel.shape)

        input_size = 1

        for x in input_shape[1:]:
            input_size *= x

        res = np.zeros((n, input_size))

        # relevance = relevance.reshape(n, -1, self.kernel.shape[1])

        # grad = relevance.dot(self.kernel.T)


        s = s.reshape(n, -1, self.kernel.shape[1])
        z = z.reshape(n, -1, self.kernel.shape[1])

        grad = (z*s).dot(self.kernel.T)
        # grad = grad.reshape(n, -1)

        for n_i in range(n):
            res[n_i][self.img_inds] += grad[n_i]
        
        res = res.reshape(a.shape)

        return a*res
        
    
    def lrp_forward(self, img:np.ndarray):
        self.last_call = img
        return self(img, lrp = True)

    def clear(self):
        self.last_call = None

class MaxPool2D:
    def __init__(self, stride, lrp_on = False):
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.lrp_on = lrp_on
    
    def __call__(self, img:np.ndarray, lrp = False):
        n, h, w, c = img.shape
        h_k, w_k = self.stride
        res = np.empty((n, h//h_k , w//w_k, c))

        if lrp: lrp_source = np.empty(img.shape)

        for n_ in range(n):
            r = pooling(img[n_], (h_k, w_k), lrp=lrp)
            if lrp:
                r, l = r

            res[n_] = r

            if lrp: lrp_source[n_] = l

        if lrp: return res, lrp_source

        return res

    def lrp_backward(self, relevance:np.ndarray):
        mode = 'avg'
        res = np.zeros_like(self.last_call)
        if mode == 'winner_takes_all':
            w = np.where(self.last_call)
            res[w[0], w[1], w[2], w[3]] = relevance.reshape(-1)
        elif 'avg':
            n,h,w,c = self.last_call.shape
            y_s, x_s = self.stride
            y_inds = np.repeat(np.arange(w//x_s), h//y_s) * y_s
            x_inds = np.tile(np.arange(h//y_s), w//x_s) * x_s
            for n_ind in range(n):
                for x_stride in range(self.stride[1]):
                    for y_stride in range(self.stride[0]):
                        x = x_inds + x_stride
                        y = y_inds + y_stride
                        res[n_ind, y, x] = relevance.reshape(-1, c)

        return res
    
    def lrp_forward(self, img:np.ndarray):
        res, self.last_call = self(img, lrp = True)
        return res

    def clear(self):
        self.last_call = None

class Flatten:
    def __init__(self, lrp_on = False):
        self.lrp_on = lrp_on

    def __call__(self, img:np.ndarray):
        n, h, w, c = img.shape
        return img.reshape(n, -1)

    def lrp_backward(self, relevance:np.ndarray):
        return relevance.reshape(self.last_call)
    
    def lrp_forward(self, img:np.ndarray):
        self.last_call = img.shape
        return self(img)

    def clear(self):
        self.last_call = None

class Dense:
    def __init__(self, weight, bias, lrp_on = False):
        self.weight = weight
        self.bias = bias
        self.lrp_on = lrp_on

    def __call__(self, img:np.ndarray):
        return (img @ self.weight) + self.bias
    
    # def lrp_backward_old(self, relevance:np.ndarray):
    #     score = self.last_call.swapaxes(0,1)*self.weight # (n, u_in, u_out)
    #     score /= score.sum(0, keepdims=True) # Normalise per weight
    #     score *= relevance
    #     score = score.sum(1) # (n, u_in) -> relevance per weight for layer above
    #     return score
    
    def lrp_backward(self, relevance:np.ndarray):
        """
            Taken from https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/tree/main
            Approximately 5 times faster than other implementation
        """
        a, w, r = self.last_call, self.weight, relevance
        z = a.dot(w) + self.bias + 1e-9   # step 1
        s = r / z               # step 2
        c = s.dot(w.T)               # step 3
        x = a*c
        return x


    def lrp_forward(self, img:np.ndarray):
        self.last_call = img
        return self(img)

    def clear(self):
        self.last_call = None

class Softmax:
    def __init__(self, axis = -1, lrp_on = False):
        self.axis = axis
        self.lrp_on = lrp_on
        self.last_call = None

    def __call__(self, x:np.ndarray):
        axis = self.axis

        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        y = y / y.sum(axis=axis, keepdims=True)
        self.last_call = y

        return y
    
    def lrp_backward(self, relevance):
        return relevance
    
    def lrp_forward(self, img:np.ndarray):
        return self(img)

    def clear(self):
        self.last_call = None

class ReLU:
    def __init__(self, lrp_on = False):
        self.last_call = None
        self.lrp_on = lrp_on

    def __call__(self, x):
        x[x<0] = 0
        return x

    def lrp_backward(self, relevance:np.ndarray):
        return relevance
    
    def lrp_forward(self, img:np.ndarray):
        return self(img)

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
    
    def lrp_backward(self):
        a = self.layers[-1].a

        for l in self.layers[:-1:-1]:
            a = l.lrp_backward(a)
        
        return a

    def lrp(self, x):
        for l in self.layers:
            x = l.lrp_forward(x)
        res = x

        res[0][1] = 0

        lrp = np.array([[1,0]])
        for l in self.layers[1:][::-1]:
            lrp = l.lrp_backward(lrp)
        
        return lrp