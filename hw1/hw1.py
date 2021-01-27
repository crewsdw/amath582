import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Read data
# Replace i by j (only need to run once)
# f_raw = open("subdata.csv", "r")
# f_raw = ''.join([i for i in f_raw]).replace('i', 'j')
# f_new = open("subdata2.csv", "w")
# f_new.writelines(f_raw)
# f_new.close()

# Read complex data
subdata = np.genfromtxt('subdata2.csv', delimiter = ',', dtype = np.complex128)

### Domain parameters
L = 10
n = 64
# Spatial domain
x2 = np.linspace(-L, L, num=n+1)
x = x2[1:n+1]
y = x
z = x
# Wavenumbers
k = 2.0*np.pi*np.fft.fftfreq(n, d=x[1]-x[0])
#ks = np.fft.fftshift(k)

# Grids
X, Y, Z = np.meshgrid(x, y, z)
#KSX, KSY, KSZ = np.meshgrid(ks, ks, ks)
KX, KY, KZ = np.meshgrid(k, k, k)

### Look at unfiltered data
sd = np.reshape(subdata[:, 0], (n,n,n))
M = np.amax(np.abs(sd))
vol = np.abs(sd)/M

### Isosurface plot at 50%
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=vol.flatten(),
    isomin=0.5,
    isomax=1,
    ))

fig.show()

### Average spectrum and determine center frequency
# Form seven step averages
avg_spec = np.zeros((n, n, n, 7))+0j
for i in range(7):
    for j in range(7):
        Unt = np.fft.fftn(np.reshape(subdata[:, 7*i + j], (n,n,n)))
        avg_spec[:,:,:,i] += Unt/7

# Center frequency = mean of where spectral density exceeds threshold (60% max)
center = np.zeros((3, 7))
for i in range(7):
    M = np.amax(np.abs(avg_spec[:,:,:,i]))
    idx = np.where(np.abs(avg_spec[:,:,:,i])/M > 0.7)
    center[0, i] = np.mean(KX[idx])
    center[1, i] = np.mean(KY[idx])
    center[2, i] = np.mean(KZ[idx])
    print(center[:,i]/(2.0*np.pi))

# For simplicity, define center as average of these window centers
c_k = np.mean(center, axis=1)/(2.0*np.pi)
c_l = np.std(center, axis=1)/(2.0*np.pi)
print('The all-time average center frequency is ' + str(c_k) + ' [1/m]')
print('The all-time st.d. is ' + str(c_l) + ' [1/m]')

### Filter and inverse transform
c_x = np.zeros((3, 49))
for i in range(7*7):
    # Discrete transform
    Unt = np.fft.fftn(np.reshape(subdata[:, i], (n,n,n)))
    # Denoise with gaussian filter
    s = 0.5 # filter strength
    mid = center[:, i//7]
    flt = np.exp(-s*((KX-mid[0])**2.0 + (KY-mid[1])**2.0 + (KZ-mid[2])**2.0))
    Unft = np.multiply(Unt, flt)
    # Reverse transform
    Unf = np.fft.ifftn(Unft)
    
    # Find center above threshold (80% max)
    idx = np.where(np.abs(Unf)/np.amax(np.abs(Unf)) > 0.8)
    c_x[0,i] = np.mean(X[idx])
    c_x[1,i] = np.mean(Y[idx])
    c_x[2,i] = np.mean(Z[idx])
    #print(c_x[:,i])


### 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(c_x[0,0], c_x[1,0], c_x[2,0], 'o--', linewidth=3, label='t=00.0 hr')
ax.plot(c_x[0,1:-1], c_x[1,1:-1], c_x[2,1:-1], 'o--', linewidth=3)
ax.plot(c_x[0,-1], c_x[1,-1], c_x[2,-1], 'o--', linewidth=3, label='t=24.5 hr')
plt.legend(loc='best')
plt.grid(True)
ax.set_xlabel(r'Position $x$')
ax.set_ylabel(r'Position $y$')
ax.set_zlabel(r'Position $z$')
plt.tight_layout()

### 2D trajectory
plt.figure()
plt.plot(c_x[0,0], c_x[1,0], 'o--', linewidth=3, label='t=00.0 hr')
plt.plot(c_x[0,1:-1], c_x[1,1:-1], 'o--', linewidth=3)
plt.plot(c_x[0,-1], c_x[1,-1], 'o--', linewidth=3, label='t=24.5 hr')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'Position $x$')
plt.ylabel(r'Position $y$')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.show()
