import numpy as np
import matplotlib.pyplot as plt
import time
# to read .mat files
from scipy.io import loadmat
#from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage

### RGB to grayscale
#def rgb2gray(rgb):
#    r, g, b = rgb[:,:,0,:], rgb[:,:,1,:], rgb[:,:,2,:]
#    gray = 0.2989*r + 0.5870*g + 0.1140*b
#    
#    return gray

### Read camera .mat files
# data11 = loadmat('data/cam1_1.mat')
# data12 = loadmat('data/cam2_1.mat')
# data13 = loadmat('data/cam3_1.mat')

# ### Turn into numpy arrays using list comprehension
# dmtrx1 = np.array([[val for val in elements]
#                    for elements in data11['vidFrames1_1']])
# dmtrx2 = np.array([[val for val in elements]
#                    for elements in data12['vidFrames2_1']])
# dmtrx3 = np.array([[val for val in elements]
#                    for elements in data13['vidFrames3_1']])

# ### Convert to grayscale
# dmtrx1g = rgb2gray(dmtrx1)
# dmtrx2g = rgb2gray(dmtrx2)
# dmtrx3g = rgb2gray(dmtrx3)

# ### Save
# with open('data/d1g.npy', 'wb') as f:
#     np.save(f, dmtrx1g)

# with open('data/d2g.npy', 'wb') as f:
#     np.save(f, dmtrx2g)

# with open('data/d3g.npy', 'wb') as f:
#     np.save(f, dmtrx3g)

### Load
ta = time.time()
with open('data/d1g.npy', 'rb') as f:
    dmtrx1g = np.load(f)

with open('data/d2g.npy', 'rb') as f:
    dmtrx2g = np.load(f)

with open('data/d3g.npy', 'rb') as f:
    dmtrx3g = np.load(f)

print('numpy time is ' + str(time.time() - ta))

### Interpolate onto the same timescale (dmtrx2 largest)
t0 = 0
tf = 1 # guess
t = np.linspace(t0, tf, num=dmtrx2g.shape[2], endpoint=True)

x = np.arange(dmtrx1g.shape[0])
y = np.arange(dmtrx1g.shape[1])
t1 = np.linspace(t0, tf, num=dmtrx1g.shape[2], endpoint=True)
#X1,Y1,T1 = (x, y, t1)
#X, Y, T = (x, y, t)

#hi = (x,y,t)
#pts = 
#print(hi)
#quit()

#dmint1 = RegularGridInterpolator((x, y, t1), dmtrx1g)

#m1 = interpn((x, y, t1), dmtrx1g, (x, y, t))

#coords = np.array([ [x[i], y[j], t[k]] for i in range(x.shape[0]) for j in range(y.shape[0]) for k in range(t.shape[0])])
#print(coords)
#quit()
#m1 = dmint1((x, y, t))
#print(m1)
#quit()

### Get them all synchronized (so annoying...!)
# First clip
s1 = 8
e1 = dmtrx1g.shape[2]
m1 = dmtrx1g[:,:,s1:e1]
# Second clip
s2 = 18
e2 = m1.shape[2]+s2
m2 = dmtrx2g[:,:,s2:e2]
# Third clip
s3 = 8
e3 = m1.shape[2]+s3
m3 = dmtrx3g[:,:,s3:e3]
# Shapes
print(m1.shape)
print(m2.shape)
print(m3.shape)

# coordinates
x = np.arange(m1.shape[0])
y = np.arange(m1.shape[1])

X,Y = np.meshgrid(x,y, indexing='ij')
#kx = np.fft.fftfreq(x.shape[0], d = x[1]-x[0])
#ky = np.fft.fftfreq(y.shape[0], d = y[1]-y[0])
#KX, KY = np.meshgrid(kx, ky, indexing='ij')
#flt = np.exp(-0.5*(KX**2.0+KY**2.0)/(0.01**2.0))

# "color filter"
m1[m1 <= 243] = 0
m2[m2 <= 240] = 0
m3[m3 <= 240] = 0

# Clear other bright objects
m1[:200,:,:] = 0
m2[:105,:,:] = 0
m2[400:,:,:] = 0
m2[:,500:,:] = 0
m3[:,:205,:] = 0

### Track the mass
# coordinates
#c1 = np.zeros((2, m1.shape[2]-2))
c1 = np.zeros((2, m1.shape[2]))
c2 = np.zeros_like(c1)
c3 = np.zeros_like(c1)
# plots
#fig, ax = plt.subplots(1,3,figsize=(15,10))
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
    
# for i in range(1,m1.shape[2]-1):
#     ### Camera 1
#     # Difference variable (avg'd over two times to denoise) (like higher order finite-difference)
#     dm = (m1[:,:,i+1] - m1[:,:,i-1])/2.0
#     # Filter
#     dmf = np.fft.fftn(dm)
#     dmff = np.multiply(flt, dmf)
#     dm_flt = np.absolute(np.fft.ifftn(dmff))
#     # Locate object
#     idx = np.where(dm_flt/np.amax(dm_flt) > 0.8)
#     c1[0,i-1] = np.mean(Y[idx])
#     c1[1,i-1] = np.mean(X[idx])
    
#     ### Camera 2
#     dm2 = (m2[:,:,i+1] - m2[:,:,i-1])/2.0
#     dmf2 = np.fft.fftn(dm2)
#     dmff2 = np.multiply(flt, dmf2)
#     dm2_flt = np.absolute(np.fft.ifftn(dmff2))
#     # Locate
#     idx = np.where(dm2_flt/np.amax(dm2_flt) > 0.8)
#     c2[0,i-1] = np.mean(Y[idx])
#     c2[1,i-1] = np.mean(X[idx])
    
#     ### Camera 3
#     dm3 = (m3[:,:,i+1] - m3[:,:,i-1])/2.0
#     dmf3 = np.fft.fftn(dm3)
#     dmff3 = np.multiply(flt, dmf3)
#     dm3_flt = np.absolute(np.fft.ifftn(dmff3))
#     # Locate
#     idx = np.where(dm3_flt/np.amax(dm3_flt) > 0.8)
#     c3[0,i-1] = np.mean(Y[idx])
#     c3[1,i-1] = np.mean(X[idx])
    
    # ax.clear()
    # ax.imshow(dm3_flt, cmap='gray')
    # ax.scatter(c3[0,i-1], c3[1,i-1])
    # plt.pause(0.05)

#print(c3.dtype)

### Kernel convolution (smooth with neighbors)
k = np.ones(5)/5 # box
#k = np.exp(-np.arange(-3, 3)**2.0)
c1s = np.array([np.convolve(c1[0,:], k, 'valid'), np.convolve(c1[1,:], k, 'valid')])
c2s = np.array([np.convolve(c2[0,:], k, 'valid'), np.convolve(c2[1,:], k, 'valid')])
c3s = np.array([np.convolve(c3[0,:], k, 'valid'), np.convolve(c3[1,:], k, 'valid')])

# Check
plt.figure()
#plt.plot(c1[0,:], c1[1,:], 'o--', label='cam 1')
plt.plot(c1s[0,:], c1s[1,:], 'o--', label='cam 1, smoothed')
#plt.plot(c2[0,:], c2[1,:], 'o--', label='cam 2')
plt.plot(c2s[0,:], c2s[1,:], 'o--', label='cam 2, smoothed')
#plt.plot(c3[0,:], c3[1,:], 'o--', label='cam 3')
plt.plot(c3s[0,:], c3s[1,:], 'o--', label='cam 3, smoothed')
plt.legend(loc='best')
plt.xlabel('Pixels x')
plt.ylabel('Pixels y')
plt.grid(True)
plt.axis([0,480,0,640])
plt.show()

### Construct data matrix
X = np.array([[c1s[0,:]],
              [c1s[1,:]],
              [c2s[0,:]],
              [c2s[1,:]],
              [c3s[0,:]],
              [c3s[1,:]]]).reshape(6, c1s.shape[1])

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
plt.plot(Y[3,:], 'o--', label='mode 3')
plt.plot(Y[4,:], 'o--', label='mode 4')
plt.plot(Y[5,:], 'o--', label='mode 5')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel('Frame')
plt.ylabel('Mode strength')
plt.title('Principal modes in baseline case')

M1f = np.fft.fft(Y[1,:])

plt.figure()
plt.plot(np.absolute(M1f))
plt.grid(True)

plt.show()
#quit()

### Downsample everything
#m1 = m1[::2, ::2, ::2]
#m2 = m2[::2, ::2, ::2]
#m3 = m3[::2, ::2, ::2]
#m1 = block_reduce(m1, block_size = (4, 4, 2), func=np.mean)
#m2 = block_reduce(m2, block_size = (4, 4, 2), func=np.mean)
#m3 = block_reduce(m3, block_size = (4, 4, 2), func=np.mean)

#print(m1.shape)
#print(m2.shape)
#print(m3.shape)

#m1s = m1.shape

#quit()

### Check it out...
# fig0, [ax0, ax1, ax2] = plt.subplots(1, 3)
# #fig0, ax0 = plt.subplots()
# #fig0, ax1 = plt.subplots()
# #fig0, ax2 = plt.subplots()
# for i in range(m3.shape[2]):
#     # Camera 1
#     ax0.clear()
#     ax0.imshow(m1[:,:,i] - np.mean(m1[:,:,i]), cmap='gray')
#     #print(i)
#     # Camera 2
#     ax1.clear()
#     ax1.imshow(m2[:,:,i], cmap='gray')
#     #print(i)
#     # ## Camera 3
#     ax2.clear()
#     ax2.imshow(m3[:,:,i], cmap='gray')
#     #print(i)
#     # Pause
#     plt.pause(0.05)
# #
# plt.show()

### Construct data matrix (rows, x*y | columns, times)
X = np.hstack((m1.reshape(m1s[0]*m1s[1], m1s[2]),
               m2.reshape(m1s[0]*m1s[1], m1s[2]),
               m3.reshape(m1s[0]*m1s[1], m1s[2])))

### Perform SVD
X -= np.mean(X)
u, s, vh = np.linalg.svd(X, full_matrices=True)

Y = np.matmul(u.T, X)

#my1 = Y[:, :m1s[2]].reshape(m1s)
# my1 = Y.reshape(m1s[0], m1s[1], 3*m1s[2])

# fig1, ax00 = plt.subplots()
# for i in range(my1.shape[2]):
#     ax00.clear()
#     ax00.imshow(my1[:,:,i], cmap='gray')
#     plt.pause(0.05)

# plt.show()

#covY = np.cov(Y)

#print(covY[:10, :10])

print(X.shape)
print(u.shape)
print(s.shape)
print(vh.shape)

plt.figure()
plt.plot(s, 'o--')

plt.figure()
#plt.plot(vh[0,:], label='Mode 0')
#plt.plot(vh[1,:], label='Mode 1')
#plt.plot(vh[2,:], label='Mode 2')
#plt.plot(vh[3,:], label='Mode 3')
#plt.plot(vh[4,:], label='Mode 4')
#plt.plot(vh[5,:], label='Mode 5')
#plt.plot(vh[6,:], label='Mode 6')
#plt.plot(vh[7,:], label='Mode 7')
plt.plot(vh[8,:], label='Mode 8')
plt.plot(vh[9,:], label='Mode 9')
plt.plot(vh[10,:], label='Mode 10')
plt.plot(vh[11,:], label='Mode 11')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()

#print(dmtrx1g.shape)
#print(dmtrx2g.shape)
#print(dmtrx3g.shape)
    
#plt.show()
