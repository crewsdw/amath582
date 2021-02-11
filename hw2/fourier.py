import numpy as np
import cmath
import numba.cuda as cuda

TPB = 8

###### Function to compute wigner function
### Wigner function
def WT_as_f(x, k, c, L):
    # fund. frequency
    k1 = 2.0*np.pi/L
    
    # Set up kernel
    blockSize = (TPB, TPB)
    numBlocksX = (x.shape[0] + blockSize[0] - 1)//blockSize[0]
    numBlocksK = (k.shape[0] + blockSize[1] - 1)//blockSize[1]
    numBlocks = (numBlocksX, numBlocksK)
    
    # output on device
    dW = cuda.device_array((x.shape[0], k.shape[0]), np.complex128)
    # input
    dC = cuda.to_device(np.ascontiguousarray(c))
    dx = cuda.to_device(x)
    dk = cuda.to_device(k)
    
    # call kernel
    wignerKernel[numBlocks, blockSize](dC, dx, dk, dW, k1)
    
    return 2.0*dW.copy_to_host()

### Wigner function kernel on GPU
@cuda.jit
def wignerKernel(dC, dx, dk, dW, k1):
    i,k = cuda.grid(2)
    
    # stay in bounds
    if (i >= dW.shape[0] or k >= dW.shape[1]):
        return
    
    # zero elements
    dW[i,k] = 0
    
    # Compute transform
    for n in range(k+1):
        f1 = dC[n]*cmath.exp(1.0j*k1*n*dx[i])
        f2 = dC[k-n]*cmath.exp(1.0j*k1*(k-n)*dx[i])
        dW[i,k] += f1.conjugate()*f2
