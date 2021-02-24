import numpy as np
import matplotlib.pyplot as plt
import time
# to read .mat files
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage

### RGB to grayscale
# def rgb2gray(rgb):
#    r, g, b = rgb[:,:,0,:], rgb[:,:,1,:], rgb[:,:,2,:]
#    gray = 0.2989*r + 0.5870*g + 0.1140*b
   
#    return gray

# ### Read camera .mat files
# data11 = loadmat('data/cam1_4.mat')
# data12 = loadmat('data/cam2_4.mat')
# data13 = loadmat('data/cam3_4.mat')

# ### Turn into numpy arrays using list comprehension
# dmtrx1 = np.array([[val for val in elements]
#                    for elements in data11['vidFrames1_4']])
# dmtrx2 = np.array([[val for val in elements]
#                    for elements in data12['vidFrames2_4']])
# dmtrx3 = np.array([[val for val in elements]
#                    for elements in data13['vidFrames3_4']])

# ### Convert to grayscale
# dmtrx1g = rgb2gray(dmtrx1)
# dmtrx2g = rgb2gray(dmtrx2)
# dmtrx3g = rgb2gray(dmtrx3)

# ### Save
# with open('data/rotate1.npy', 'wb') as f:
#     np.save(f, dmtrx1g)

# with open('data/rotate2.npy', 'wb') as f:
#     np.save(f, dmtrx2g)

# with open('data/rotate3.npy', 'wb') as f:
#     np.save(f, dmtrx3g)

### Load files
ta = time.time()
with open('data/rotate1.npy', 'rb') as f:
    dm1 = np.load(f)

with open('data/rotate2.npy', 'rb') as f:
    dm2 = np.load(f)

with open('data/rotate3.npy', 'rb') as f:
    dm3 = np.load(f)

print('numpy time is ' + str(time.time() - ta))

# print(dm1.shape)
# print(dm2.shape)
# print(dm3.shape)

### Get them synchronized (so annoying!!)
# first clip
s1 = 0
e1 = dm1.shape[2]#+s1
m1 = dm1[:,:,s1:e1]
# second clip
s2 = 8
e2 = m1.shape[2]+s2
m2 = dm2[:,:,s2:e2]
# third clip
s3 = 0
e3 = m1.shape[2]+s3
m3 = dm3[:,:,s3:e3]
# Shapes
#print(m1.shape)
#print(m2.shape)
#print(m3.shape)

# "color filter"
m1[m1 <= 243] = 0
m2[m2 <= 240] = 0
m3[m3 <= 230] = 0

# Clear other bright objects
m1[:220,:,:] = 0
m2[:140,:,:] = 0
m2[:,:210,:] = 0
m2[:,550:,:] = 0
m3[:,:240,:] = 0

# Coords
x = np.arange(m1.shape[0])
y = np.arange(m2.shape[1])
X,Y = np.meshgrid(x,y, indexing='ij')

# mean pos
c1 = np.zeros((2, m1.shape[2]))
c2 = np.zeros_like(c1)
c3 = np.zeros_like(c2)
#fig, ax = plt.subplots(1,3, figsize=(15,10))
for i in range(m3.shape[2]):
    # cam1
    #ax[0].clear()
    #ax[0].imshow(m1[:,:,i], cmap='gray')
    # cam2
    #ax[1].clear()
    #ax[1].imshow(m2[:,:,i], cmap='gray')
    # cam3
    #ax[2].clear()
    #ax[2].imshow(m3[:,:,i], cmap='gray')
    #plt.pause(0.05)

    # Get mean positions
    # cam1
    c1[0,i] = np.mean(Y[np.where(m1[:,:,i]>0)])
    c1[1,i] = np.mean(X[np.where(m1[:,:,i]>0)])
    # cam2
    c2[0,i] = np.mean(Y[np.where(m2[:,:,i]>0)])
    c2[1,i] = np.mean(X[np.where(m2[:,:,i]>0)])
    # cam3
    c3[0,i] = np.mean(Y[np.where(m3[:,:,i]>0)])
    c3[1,i] = np.mean(X[np.where(m3[:,:,i]>0)])
    
# Check coordinates
plt.figure()
plt.plot(c1[0,:], c1[1,:], 'o--', label='cam 1')
#0plt.plot(c1s[0,:], c1s[1,:], 'o', label='cam 1, smoothed')
plt.plot(c2[0,:], c2[1,:], 'o--', label='cam 2')
#plt.plot(c2s[0,:], c2s[1,:], 'o', label='cam 2, smoothed')
plt.plot(c3[0,:], c3[1,:], 'o--', label='cam 3')
#plt.plot(c3s[0,:], c3s[1,:], 'o', label='cam 3, smoothed')
plt.legend(loc='best')
plt.grid(True)
plt.axis([0,480,0,640])
plt.show()

### Construct data matrix
X = np.array([[c1[0,:]],
              [c1[1,:]],
              [c2[0,:]],
              [c2[1,:]],
              [c3[0,:]],
              [c3[1,:]]]).reshape(6, c1.shape[1])

### SVD it
X -= np.mean(X)
u, s, vh = np.linalg.svd(X, full_matrices=True)
# Principal component projection
Y = np.matmul(u.T, X)
# Check it out...
plt.figure()
plt.plot(Y[0,:], 'o--', label='mode 0')
plt.plot(Y[1,:], 'o--', label='mode 1')
plt.plot(Y[2,:], 'o--', label='mode 2')
#plt.plot(Y[3,:], 'o--', label='mode 3')
#plt.plot(Y[4,:], 'o--', label='mode 4')
#plt.plot(Y[5,:], 'o--', label='mode 5')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Frame')
plt.ylabel('Mode strength')
plt.title('Principal modes of rotating oscillator')

### FFT the PC proj.
M1f = np.fft.fft(Y[1,:])
M2f = np.fft.fft(Y[2,:])

plt.figure()
plt.semilogy(np.absolute(M1f), label='Mode 1')
plt.semilogy(np.absolute(M2f), label='Mode 2')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Fourier mode')
plt.ylabel('Amplitude (a.u.)')
plt.tight_layout()

plt.show()
