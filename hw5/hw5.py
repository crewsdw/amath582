import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import gc # garbage collector to delete big stuff

# To make movies
import matplotlib.animation as animation

# To read movies
import av

def rgb2gray(rgb):
   r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
   gray = 0.2989*r + 0.5870*g + 0.1140*b
   
   return gray

### Open, convert, and save (Run once per file)
#Open file
#v = av.open('monte_carlo_low.mp4')
v = av.open('ski_drop_low.mp4')

fps = 60.0

# Convert frames to images
#vid = np.zeros((380, 1112, 1872)) # monte carlo size
#vid = np.zeros((380, 540, 960)) # monte carlo low size
# vid = np.zeros((454, 540, 960))
# counter = 0
# for packet in v.demux():
#     for frame in packet.decode():
#         #print(frame)
#         img = frame.to_image()
#         vid[counter,:,:] = rgb2gray(np.asarray(img))
#         counter += 1

### Save, faster to save it now and load as np array
#with open('data/montecarlo_low.npy', 'wb') as f:
#   np.save(f, vid)

# with open('data/skidrop_low.npy', 'wb') as f:
#    np.save(f, vid)

### Load
with open('data/montecarlo_low.npy', 'rb') as f:
   vid = np.load(f)

# with open('data/skidrop_low.npy', 'rb') as f:
#    vid = np.load(f)

#vid = vid[:,:,230:730]
   
### Reshape into data matrix, frames x pixels
X_T = vid[:-1,:,:].reshape(vid.shape[0]-1, vid.shape[1]*vid.shape[2])
XP_T = vid[1:,:,:].reshape(vid.shape[0]-1, vid.shape[1]*vid.shape[2])
# Flip for DMD convention, each frame is a column
X = X_T.T
XP = XP_T.T

del X_T
del XP_T
gc.collect()

# print('Beginning SVD...')
# u, s, vh = np.linalg.svd(X, full_matrices=False)

# ### Save SVD matrices
# with open('data/skidrop_low_svd_u.npy', 'wb') as f:
#    np.save(f, u)

# with open('data/skidrop_low_svd_s.npy', 'wb') as f:
#    np.save(f, s)

# with open('data/skidrop_low_svd_vh.npy', 'wb') as f:
#    np.save(f, vh)

# quit()

### Load SVD matrices
# with open('data/skidrop_low_svd_u.npy', 'rb') as f:
#     u = np.load(f)

# with open('data/skidrop_low_svd_s.npy', 'rb') as f:
#     s = np.load(f)

# with open('data/skidrop_low_svd_vh.npy', 'rb') as f:
#     vh = np.load(f)

with open('data/montecarlo_low_svd_u.npy', 'rb') as f:
    u = np.load(f)

with open('data/montecarlo_low_svd_s.npy', 'rb') as f:
    s = np.load(f)

with open('data/montecarlo_low_svd_vh.npy', 'rb') as f:
    vh = np.load(f)

### Truncate modes
print('Files loaded...')

v = vh.T

cutoff = 100#200
ur = u[:, :cutoff]
sr = s[:cutoff]
vr = v[:, :cutoff]

### Koopman operator: low-rank projection
Atilde = np.dot(ur.T, np.dot(XP, np.dot(vr, np.diag(1.0/sr))))

### Eigenvalue problem for Atilde
w, va = np.linalg.eig(Atilde)

### Koopman operator: lift projection to dynamic modes
phi = np.dot(XP, np.dot(vr, np.dot(np.diag(1.0/sr), va)))#np.dot(ur, va)

### Find initial condition among modes
b = np.dot(np.linalg.pinv(phi), X[:,0])

### Full time-zero construction
Xrec = np.dot(phi, b)

# Vid size
size = (vid.shape[1], vid.shape[2])

### DMD frequencies
# Time
t = np.arange(vid.shape[0])/fps
dt = t[1] - t[0]

# Freqs
om = np.log(w)/dt

### Sort phi, b, and omega
idx = np.argsort(np.abs(om))
Phi = phi[:, idx]
Om = om[idx]

### Get lowest frequencies
sidx = np.where(np.absolute(Om) < 2) # mc, 2 | ski drop 1
Ds = np.diag(Om[sidx])
Phis = Phi[:, sidx][:,0,:]

# Background b
bs = np.zeros((Phis.shape[1], vid.shape[0])) + 0j
bs[:,:-1] = np.dot(np.linalg.pinv(Phis), X)
bs[:,-1] = np.dot(np.linalg.pinv(Phis), XP[:,-1])

print('DMD finished...')

### Delete stuff
del u
del s
del vh
del phi
del Phi
del Atilde
gc.collect()

plt.figure()
plt.plot(np.real(Om), 'o--', label='real')
plt.plot(np.imag(Om), 'o--', label='imag')
plt.plot(np.absolute(om), 'o--', label='abs')
plt.plot(np.absolute(Om), 'o--', label='abs sorted')
plt.xlabel('DMD mode number')
plt.ylabel('Frequency')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(np.real(Om), np.imag(Om), 'o')
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.tight_layout()

# Background recons
Brc = np.dot(Phis, bs)
bkgrnd = Brc.T.reshape(vid.shape)

#Brc = np.array(
plt.figure()
plt.imshow(np.absolute(bkgrnd[200,:,:]), cmap='gray')
plt.title('Background reconstruction')
plt.colorbar()

plt.show()

print('Reconstruction completed...')

frames = vid - np.absolute(bkgrnd)
#frames[frames < 0] = 0 # add back in zero values
frames[frames < 0] = np.absolute(frames[frames<0]) # absolute value of negatives 
frames[frames < 50] = 50 # background filter...

fig, ax = plt.subplots()
anims = []
for i in range(t.shape[0]):
   this_im = ax.imshow(frames[i,:,:], cmap='gray', vmin=0, vmax=np.amax(frames))
   anims.append([this_im])

an = animation.ArtistAnimation(fig, anims, interval=50, blit=True, repeat_delay=1000)
an.save('montecarlo_nbg3.mp4', fps=60)

quit()
