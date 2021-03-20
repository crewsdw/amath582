# Main libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
from scipy import interpolate

# My libraries
import funcs as f
import datamgmt as dm
import basis as b
import reference as r
import kernels as dg
import quasilinear as ql
import elliptic as el
import filters as fl
import kernels1D as dg1D

### Handy functions
# Flatten excluding ghost cells
def flat(ar):
    return ar[1:-1,:].flatten()

# geometry set-up
def geo_setup(ginfo, n):
    Nx = int(ginfo[0])
    xMin = ginfo[1]
    xMax = ginfo[2]
    xnodes = np.array(b.getNodes(n))
    iso_nx = (xnodes + 1)/2
    return Nx, xMin, xMax, xnodes, iso_nx

# colorbars
def cb(ar):
    return np.linspace(np.amin(ar), np.amax(ar), num=150)

### Full Wigner function
def WT_fs(x, k, c, L, num0, num_modes):
    modes = np.arange(-num_modes+1, num_modes)
    #print(modes)
    # fund. freq.
    k1 = 2.0*np.pi/L
    # conjugate coefficients
    cc = np.conj(c)
    # compute wig. func.
    w = np.zeros((x.shape[0], k.shape[0])) + 0j
    ### Discrete frequency range, outermost loop
    lidx = -num0+1
    uidx = num0
    #print(lidx)
    #print(uidx)
    
    for l in range(lidx, uidx):
        # Shift for array index
        l_idx = l - lidx
        ### Interference frequency range, inner loop
        for n in modes:
            if (modes[0] < l-n < modes[-1]):
                n_idx = n - (-num_modes+1)
                ln_idx = (l-n) - (-num_modes+1)
                # Check indices
                cn = c[n_idx]
                cln = c[ln_idx]
                # First fourier coefficient
                fn = cn*np.exp(1.0j*k1*n*x)
                # second fourier coefficient
                fln = cln*np.exp(1.0j*k1*(l-n)*x)
                # wigner series term
                w[:,l_idx] += np.multiply(np.conj(fn), fln)
    
    return w*L/np.pi

### Phase space shift function
def phase_space_shift(pdf_rs, KX, KV, t):
    # FFT distribution
    pdf_kv = np.fft.fft2(pdf_rs)
    pdf_b = np.real(np.fft.ifft2(pdf_kv))
    #kf2 = np.fft.fftshift(np.fft.fftfreq(xbar[1:-1].shape[0], d=xbar[1]-xbar[0]))
    
    # Phase space translation phase shift
    phase = np.exp(1j*2.0*np.pi*group*(KV + KX*t))
    pdf_kv_t = np.multiply(phase, np.fft.fftshift(pdf_kv))
    pdf_b_t = np.real(np.fft.ifft2(np.fft.fftshift(pdf_kv_t)))
    
    return pdf_b_t

# Global filename
#folder = 'autumn20/'
#fname = '1000L_150x150_8n'
folder = 'winter20/'
fname = '1000L_bot_wprotons'
#fname = 'lang_decay2'

###################################################################################################
###### Read data files
###################################################################################################
# Initialize data class
run_data = dm.run_data(folder, fname, [0, 0, 0, 0], pdf2=False) # null entry
pdf, pot, den, t = run_data.readData()
#pdf, pdf2, pot, den, t = run_data.readData()

# Get run info
ginfo, tinfo, rvals = run_data.readInfo()

### Run info
n = int(ginfo[0,3]) # x nodes
m = int(ginfo[1,3]) # y nodes
k = int(tinfo[2]) # temporal order

###################################################################################################
###### Set up reference parameters
###################################################################################################
# inputs: n0 (p), T0 (p), TeTi, mFrac (me_desired/me_true)
ref = r.reference(rvals[0], rvals[1], 1.0, 1.0)#1.0)
# get thermal velocity, electron acceleration coefficient, charge density coefficient
vti, vte, eMult, iMult, cMult = ref.getParameters()
ometau = ref.getOmetau()
Ae = ref.getAe()
dp = ref.getdp()

###################################################################################################
###### Create grid
###################################################################################################
###### Domain parameters
# x
Nx, xMin, xMax, xnodes, iso_nx = geo_setup(ginfo[0,:], n)
# v electrons
Nv, vMin, vMax, vnodes, iso_nv = geo_setup(ginfo[1,:], m)
# v protons
Nvp, vpMin, vpMax, vpnodes, iso_nvp = geo_setup(np.array([150, -8*vti, 8*vti]), m)
# t
Nt = t.shape[0]
final_t = t[-1]
dt = t[1]-t[0]

#print(Nt)
#quit()

###### Create grids
x = f.createGrid1D(xMin, xMax, Nx, iso_nx)
v = f.createGrid1D(vMin, vMax, Nv, iso_nv)
vp = f.createGrid1D(vpMin, vpMax, Nvp, iso_nvp)
dx = x[2,0] - x[1,0]
dv = v[2,0] - v[1,0]
dvp = vp[2,0] - vp[1,0]
L = xMax - xMin
# middles
xbar = np.array([(x[i,-1]+x[i,0])/2.0 for i in range(x.shape[0])])
vbar = np.array([(v[i,-1]+v[i,0])/2.0 for i in range(v.shape[0])]) # +1, ghost cells

# Even grids
xfine = np.linspace(xMin, xMax, n*Nx//5)
vfine = np.linspace(vMin, vMax, m*Nv//5)

###### Get Jacobians
Jx = b.getJacobian(n, dx)
Jv = b.getJacobian(m, dv)
Jvp = b.getJacobian(m, dvp)

J = np.array([Jx, Jv])
Jp = np.array([Jx, Jvp])

###### Make flat grids and mesh grid
# flat grids
xf = flat(x)#x[1:-1,:].flatten()
vf = flat(v)#v[1:-1,:].flatten()
xf2 = x[1:-1,:-1].flatten()
vf2 = v[1:-1,:-1].flatten()

vpf = flat(vp)#vp[1:-1,:].flatten()
# meshgrids of flat grids
XF,VF = np.meshgrid(xf, vf, indexing='ij')
XF2, VF2 = np.meshgrid(xf2, vf2, indexing='ij')

XP, VP = np.meshgrid(xf, vpf, indexing='ij')
XX,TT = np.meshgrid(xf, t)#, indexing='ij')
#XB, VB = np.meshgrid(xbar[1:-1], vbar[1:-1])
XB, VB = np.meshgrid(xfine, vfine)

###################################################################################
##################### AVERAGE DISTRIBUTION ANALYSIS
###################################################################################
avg_pdf = np.zeros((Nt, v.shape[0], v.shape[1]))
#fig, ax = plt.subplots()
for i in range(Nt):
    avg_pdf[i,:,:] = f.avgPDF(pdf[i, 1:-1, :, :], n, m, Jv, L)

# SVD it
X_T = avg_pdf[:-1,:,:].reshape((Nt-1, v.shape[0]*v.shape[1]))
XP_T = avg_pdf[1:,:,:].reshape((Nt-1, v.shape[0]*v.shape[1]))

X = X_T.T
XP = XP_T.T

u, s, vh = np.linalg.svd(X, full_matrices=False)

plt.figure()
plt.semilogy(s, 'o--')
plt.xlabel('Mode number')
plt.ylabel('Singular values')
plt.grid(True)
plt.tight_layout()

# Koopman operator (1D so only 1216x1216)
A = np.dot(XP, np.linalg.pinv(X))
# Similarity transform
At = np.dot(u.T, np.dot(A, u))
# Eigenvalue problem
w, eigv = np.linalg.eig(At)
# Sort by absolute value
om = np.log(w)/dt
idx = np.argsort(np.abs(om))
Om = om[idx]
Eigv = eigv[:,idx]

Phi = np.dot(XP, Eigv)
Phiv = Phi.reshape((v.shape[0], v.shape[1], Nt-1))

plt.figure()
plt.plot(flat(v), flat(avg_pdf[0,:,:]), 'o--', label='initial distribution')
plt.plot(flat(v), flat(np.absolute(Phiv[:,:,0])), 'o--', label='zeroth mode')
plt.plot(flat(v), np.absolute(flat(Phiv[:,:,1])), 'o--', label='first mode')
plt.xlabel(r'Velocity')
plt.ylabel(r'Distribution f(v)')
#for i in range(5):
#    plt.plot(flat(v), np.absolute(flat(Phi[:,:,i])), 'o--', label='mode' + str(i))
#plt.plot(flat(v), np.absolute(flat(Phi[:,:,0])), 'o--', label='zero mode')
#
#plt.plot(flat(v), np.absolute(flat(Phi[:,:,2])), 'o--', label='second mode')
plt.grid(True)
plt.legend(loc='best')

### Modal coefficients
b0 = np.dot(np.linalg.pinv(Phi), X[:,0])
bs = np.zeros((Phi.shape[1], Nt)) + 0j
bs[:,:-1] = np.dot(np.linalg.pinv(Phi), X)
bs[:,-1] = np.dot(np.linalg.pinv(Phi), XP[:,-1])
#bt = np.array([np.dot(np.diag(np.exp(Om*t[i])), b0) for i in range(t.shape[0])])

# plt.figure()
# for i in range(Nt):
#     plt.plot(flat(v), flat(avg_pdf[i,:,:]), 'o--', label='mode ' + str(i))
# plt.legend(loc='best')

plt.figure()
plt.semilogy(np.absolute(bs[:,0]), 'o', label='Initial condition')
plt.semilogy(np.absolute(bs[:,15]), 'o', label='Growing instability')
plt.semilogy(np.absolute(bs[:,55]), 'o', label='Heating and flattening') 
plt.semilogy(np.absolute(bs[:,-1]), 'o', label='Saturation')
plt.xlabel('Dynamic mode number')
plt.ylabel('Amplitude [arb. units]')
#plt.plot(t, np.absolute(bs[1,:]), 'o--', label='Mode 1')
#plt.plot(t, np.absolute(bs[2,:]), 'o--', label='Mode 2')
plt.legend(loc='best')
plt.grid(True)

X0 = np.dot(Phi, b0).reshape(v.shape[0], v.shape[1])
X50 = np.dot(Phi, bs[:,50]).reshape(v.shape[0], v.shape[1])

plt.figure()
plt.plot(flat(v), flat(np.real(X0)), 'o--', label='t0 reconstruction')
plt.plot(flat(v), flat(avg_pdf[0,:,:]), 'o--', label='initial distribution')
plt.plot(flat(v), flat(np.real(X50)), 'o--', label='t50 reconstruction')
plt.plot(flat(v), flat(avg_pdf[50,:,:]), 'o--', label='t50 distribution')
plt.legend(loc='best')
plt.grid(True)

### Koopman eigenvalues
plt.figure()
plt.plot(np.absolute(Om), 'o--')
plt.xlabel('Mode number')
plt.ylabel('Absolute value of eigenfrequency')
plt.grid(True)

plt.figure()
plt.plot(np.real(Om), np.imag(Om), 'o')
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title(r'Eigenvalue spectrum of Koopman operator for $\langle f\rangle_v$')
plt.tight_layout()
plt.show()

###################################################################################
##################### AVERAGE DISTRIBUTION ANALYSIS
###################################################################################
### Remove translational invariance
T = np.mean(b.moment(pdf[0, 1:-1, 1:-1, :], v, 2, Jv))
print(T)
group = (3*T)**0.5/np.sqrt(1.0 + 1.0/(3*(0.25)**2))
print(group)

### Frequency fun
lim = 200#30
k0 = 2.0*np.pi/L
#wig = np.zeros((Nt, xbar.shape[0], 2*lim-1)) + 0j
#fig, ax = plt.subplots()
den_ds = np.zeros((Nt, xf.shape[0]))
# Resampled and resampled/shifted distributions
pdf_rs = np.zeros((Nt, XB.shape[0], XB.shape[1]))
pdf_rs_s = np.zeros((Nt, XB.shape[0], XB.shape[1]))
u0_rs_s = np.zeros_like(pdf_rs_s)

# Wavenumbers
kx = np.fft.fftfreq(xfine.shape[0], d=xfine[1]-xfine[0])
kx = np.fft.fftshift(kx)
kv = np.fft.fftfreq(vfine.shape[0], d=vfine[1]-vfine[0])
kv = np.fft.fftshift(kv)
KX, KV = np.meshgrid(kx, kv)

## Transform velocity
vt = group
#vbart = vbar - vt
vfinet = vfine - vt

#XB2, VB2 = np.meshgrid(xbar[1:-1], vbart[1:-1])
XB2, VB2 = np.meshgrid(xfine, vfinet)

# Fill timesteps
for i in range(Nt):
    ### Interp
    #Upts = f.flat2Ds(pdf[i, 1:-1, 1:-1, :], n, m)
    #fint = interpolate.RectBivariateSpline(xf2, vf2, Upts)
    #pdf_rs[i,:,:] = fint.ev(XB, VB)
    
    ### Compute frequency spectrum
    # electron density spectrum
    n0 = b.moment(pdf[i, 1:-1, 1:-1, :], v, 0, Jv)
    wvs, amps = b.FFS(n0-np.mean(n0), xf, xbar[1:-1], 1.0/Jx, L, lim, cmplx_2side=True)
    
    #pdf_rs_s[i,:,:] = phase_space_shift(pdf_rs[i,:,:], KX, KV, t[i])
    #u0_rs_s[i,:,:] = phase_space_shift(u0_rs, KX, KV, t[i])
    
    # Doppler shift
    phase = np.exp(1j*wvs*group*t[i])
    amps2 = np.multiply(amps, phase)
    den_ds[i,:] = np.real(b.sumFourier(amps2, wvs, xf))
    # Wigner distribution
    #wig[i,:,:] = WT_fs(xbar, wvs/2.0, amps2, L, lim, lim)
    
# plt.figure()
# plt.contourf(XX, TT, den_ds, cb(den_ds))
# plt.title('Langmuir group velocity frame, v={:0.2f}'.format(group))
# plt.xlabel('Position')
# plt.ylabel('Time')
# plt.tight_layout()
# plt.colorbar()
# plt.show()

# Save Wigner because it takes a while to calculate
# with open('wigner.npy', 'wb') as f:
#    np.save(f, wig)

# Open Wigner
with open('wigner.npy', 'rb') as f:
    wig = np.load(f)

### Subtract mean
#den_xf = den.reshape(den.shape[0], den.shape[1]*den.shape[2]) - 1.0

### DMD analysis
# SVD it
Y_T = den_ds[:-1,:]
YP_T = den_ds[1:,:]

Y = Y_T.T
YP = YP_T.T

u, s, vh = np.linalg.svd(Y, full_matrices=False)

# Singular values
plt.figure()
plt.semilogy(s, 'o--')
plt.xlabel('Mode number')
plt.ylabel(r'Singular values of density n(x) in rest frame')
plt.grid(True)
plt.tight_layout()

### DMD analysis
# Koopman operator (1D so only 1200x1200)
A = np.dot(YP, np.linalg.pinv(Y))
# Similarity transform to time domain
At = np.dot(u.T, np.dot(A, u))
# Eigenvalue problem
w, eigv = np.linalg.eig(At)
# Sort by absolute value
om = np.log(w)/dt
idx = np.argsort(np.abs(om))
Om = om[idx]
Eigv = eigv[:,idx]

Phi = np.dot(YP, Eigv)
Phix = Phi.reshape((x[1:-1,:].shape[0], x[1:-1,:].shape[1], Nt-1))

plt.figure()
plt.plot(flat(x), den_ds[0,:], 'o--', label='initial density')
plt.plot(flat(x), np.real(Phix[:,:,0]).flatten(), 'o--', label='zeroth mode')
plt.plot(flat(x), np.real(Phix[:,:,1]).flatten(), 'o--', label='first mode')
plt.plot(flat(x), np.real(Phix[:,:,2]).flatten(), 'o--', label='second mode')
plt.legend(loc='best')
plt.xlabel(r'Position')
plt.ylabel(r'Density fluctuation n(x)')

plt.figure()
plt.plot(np.absolute(Om), 'o--')
plt.xlabel('Mode number')
plt.ylabel('Absolute value of eigenfrequency')
plt.grid(True)

plt.figure()
plt.plot(np.real(Om), np.imag(Om), 'o')
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title(r'Eigenvalue spectrum of Koopman operator for density fluctuation $\delta n(x)$')
plt.tight_layout()

### Modal coefficients
b0 = np.dot(np.linalg.pinv(Phi), Y[:,0])
bs = np.zeros((Phi.shape[1], Nt)) + 0j
bs[:,:-1] = np.dot(np.linalg.pinv(Phi), Y)
bs[:,-1] = np.dot(np.linalg.pinv(Phi), YP[:,-1])

plt.figure()
plt.semilogy(np.absolute(bs[:,0]), 'o', label='Random initial condition')
plt.semilogy(np.absolute(bs[:,10]), 'o', label='Landau damping of most modes')
plt.semilogy(np.absolute(bs[:,40]), 'o', label='Growth of wavepackets')
#plt.semilogy(np.absolute(bs[:,45]), 'o', label='Further growth')
plt.semilogy(np.absolute(bs[:,-1]), 'o', label='Saturated distribution')
plt.xlabel('Dynamic mode number')
plt.ylabel('Amplitude [arb. units]')
#plt.plot(t, np.absolute(bs[1,:]), 'o--', label='Mode 1')
#plt.plot(t, np.absolute(bs[2,:]), 'o--', label='Mode 2')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()

###################################################################################
##################### WIGNER ANALYSIS
###################################################################################
# Flatten
wig_f = wig.reshape((Nt, wig.shape[1]*wig.shape[2]))
# DMD arrays
Z = wig_f[:-1,:].T
ZP = wig_f[1:,:].T
# SVD of Z
u, s, vh = np.linalg.svd(Z, full_matrices=False)

print(u.shape)
print(s.shape)
print(vh.shape)

# Singular values
plt.figure()
plt.semilogy(s, 'o--')
plt.xlabel('Mode number')
plt.ylabel(r'Singular values [arb. units]')
plt.title('Wigner function $\psi(x,k)$ in rest frame')
plt.grid(True)
plt.tight_layout()

### Truncated Koopman operator
cutoff=-1
v = vh.T
ur = u[:, :cutoff]
sr = s[:cutoff]
vr = v[:, :cutoff]

# Koopman operator : low-rank projection
Atilde = np.dot(ur.T, np.dot(ZP, np.dot(vr, np.diag(1.0/sr))))
# Eigenvalue problem for projected operator
w, eigv = np.linalg.eig(Atilde)
# Sort by absolute value
om = np.log(w)/dt
idx = np.argsort(np.abs(om))
Om = om[idx]
Eigv = eigv[:,idx]
# Lift projection to dynamic modes
Phi = np.dot(ZP, np.dot(vr, np.dot(np.diag(1.0/sr), Eigv)))

Phi_ps = Phi.reshape((wig.shape[1], wig.shape[2], Phi.shape[1]))

WX, WK = np.meshgrid(xbar, wvs/2.0/k0, indexing='ij')
plt.figure()
m0 = np.real(Phi_ps[:,:,0])
plt.contourf(WX, WK, m0, cb(m0))
plt.colorbar()
plt.xlabel(r'Position $x/\lambda_D$')
plt.ylabel(r'Wavenumber $k\lambda_D$')
plt.title(r'Zeroth mode of Wigner function $\psi(x,k)$')
plt.tight_layout()

m1 = np.real(Phi_ps[:,:,1])
plt.figure()
plt.contourf(WX, WK, m1, cb(m1))
plt.colorbar()
plt.xlabel(r'Position $x/\lambda_D$')
plt.ylabel(r'Wavenumber $k\lambda_D$')
plt.title(r'First mode of Wigner function $\psi(x,k)$')
plt.tight_layout()

m25 = np.real(Phi_ps[:,:,25])
plt.figure()
plt.contourf(WX, WK, m25, cb(m25))
plt.colorbar()
plt.xlabel(r'Position $x/\lambda_D$')
plt.ylabel(r'Wavenumber $k\lambda_D$')
plt.title(r'Twenty-fifth mode of Wigner function $\psi(x,k)$')
plt.tight_layout()

plt.figure()
plt.plot(np.absolute(Om), 'o--')
plt.xlabel('Mode number')
plt.ylabel('Absolute value of eigenfrequency')
plt.grid(True)

plt.figure()
plt.plot(np.real(Om), np.imag(Om), 'o')
plt.grid(True)
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title(r'Eigenvalue spectrum of Wigner function $\psi(x,k)$ Koopman operator')
plt.tight_layout()

### Modal coefficients
#b0 = np.dot(np.linalg.pinv(Phi), Z[:,0])
bs = np.zeros((Phi.shape[1], Nt)) + 0j
bs[:,:-1] = np.dot(np.linalg.pinv(Phi), Z)
bs[:,-1] = np.dot(np.linalg.pinv(Phi), ZP[:,-1])

plt.figure()
plt.semilogy(np.absolute(bs[:,0]), 'o', label='Random initial condition')
plt.semilogy(np.absolute(bs[:,10]), 'o', label='Landau damping of most modes')
plt.semilogy(np.absolute(bs[:,40]), 'o', label='Growth of wavepackets')
#plt.semilogy(np.absolute(bs[:,45]), 'o', label='Further growth')
plt.semilogy(np.absolute(bs[:,-1]), 'o', label='Saturated distribution')
plt.xlabel('Dynamic mode number')
plt.ylabel('Amplitude [arb. units]')
#plt.plot(t, np.absolute(bs[1,:]), 'o--', label='Mode 1')
#plt.plot(t, np.absolute(bs[2,:]), 'o--', label='Mode 2')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

plt.show()

#ax.contourf(WX, WK, np.real(vhw[i,:].reshape(wig.shape[1], wig.shape[2])), cb(np.real(vhw[i,:])))



# ### Sampling indices
# idxs = np.array([0, Nt//4, Nt//2, 3*Nt//4, 4*Nt//5, -1])
# # times
# t_d = t[idxs]

# ### SVD the distribution function
# # Subtract initial state
# beams = 2 # number of beams

# Bump on tail (BOT) parameters
chi = 0.05 # BOT fraction parameter
alpha = 1.0 # BOT energy parameter
frac1 = 1.0/(1.0 + chi) # main fraction
frac2 = chi/(1.0 + chi) # BOT fraction
mShift = 0.0 # main mean shift
bShift = 5.0*vte # bump mean shift
vtb = alpha*chi**(1.0/3.0)*bShift # bump thermal velocity

z = np.array([0])
u0 = f.createBumpIC(x, v, vte, vtb, frac1, frac2, mShift, bShift, z, z, 'bump-on-tail')

u0f = f.flat2Ds(u0[1:-1,1:-1,:], n, m)
fint = interpolate.RectBivariateSpline(xf2, vf2, u0f)
u0_rs = fint.ev(XB, VB)

# pdf_ng = pdf[:,1:-1,1:-1,:] - u0[1:-1, 1:-1, :]
# X = pdf_ng.reshape(pdf_ng.shape[0], pdf_ng.shape[1]*pdf_ng.shape[2]*pdf_ng.shape[3])

# u, s, vh = np.linalg.svd(X, full_matrices=False)

# #print(u.shape)
# #print(s.shape)
# #print(vh.shape)

# plt.figure()
# plt.semilogy(s, 'o')
# plt.xlabel('Mode number')
# plt.ylabel('Spectral amplitude [arb. units]')
# plt.grid(True)

# fig, ax = plt.subplots()

# for i in range(10):
#     vhi = vh[i,:]
#     vhirs = vhi.reshape(pdf_ng.shape[1], pdf_ng.shape[2], pdf_ng.shape[3])
#     ax.clear()
#     ax.contourf(XF, VF, f.flat2D(vhirs, n, m), cb(vhirs))
#     plt.pause(2)

# #plt.figure()


# #print(pdf_ng.shape)
# #print(X.shape)
# plt.show()
# quit()


pot_ng = pot[:,1:-1,:]
pot_xf = pot_ng.reshape(pot.shape[0], pot_ng.shape[1]*pot.shape[2])
#u, s, vh = np.linalg.svd(den_xf)

# ### Density plots
# plt.figure()
# plt.plot(s, 'o')
# plt.ylabel('Singular values')

plt.figure()
plt.contourf(XX, TT, den_xf, cb(den_xf))
plt.colorbar()

# #plt.figure()
# #plt.plot(xf, vh[0,:], 'o--')
# #plt.plot(xf, vh[1,:], 'o--')
# #plt.plot(xf, vh[2,:], 'o--')


plt.show()
#quit()


################################################################################



### Frequency fun
lim = 200#30
k0 = 2.0*np.pi/L
#wig = np.zeros((Nt, xbar.shape[0], 2*lim-1)) + 0j
fig, ax = plt.subplots()
den_ds = np.zeros((Nt, xf.shape[0]))
# Resampled and resampled/shifted distributions
pdf_rs = np.zeros((Nt, XB.shape[0], XB.shape[1]))
pdf_rs_s = np.zeros((Nt, XB.shape[0], XB.shape[1]))
u0_rs_s = np.zeros_like(pdf_rs_s)

# Wavenumbers
kx = np.fft.fftfreq(xfine.shape[0], d=xfine[1]-xfine[0])
kx = np.fft.fftshift(kx)
kv = np.fft.fftfreq(vfine.shape[0], d=vfine[1]-vfine[0])
kv = np.fft.fftshift(kv)
KX, KV = np.meshgrid(kx, kv)

## Transform velocity
vt = group
#vbart = vbar - vt
vfinet = vfine - vt

#XB2, VB2 = np.meshgrid(xbar[1:-1], vbart[1:-1])
XB2, VB2 = np.meshgrid(xfine, vfinet)

# Fill timesteps
for i in range(Nt):
    ### Interp
    Upts = f.flat2Ds(pdf[i, 1:-1, 1:-1, :], n, m)
    fint = interpolate.RectBivariateSpline(xf2, vf2, Upts)
    pdf_rs[i,:,:] = fint.ev(XB, VB)
    
    ### Compute frequency spectrum
    # electron density spectrum
    n0 = b.moment(pdf[i, 1:-1, 1:-1, :], v, 0, Jv)
    wvs, amps = b.FFS(n0-np.mean(n0), xf, xbar[1:-1], 1.0/Jx, L, lim, cmplx_2side=True)
    
    pdf_rs_s[i,:,:] = phase_space_shift(pdf_rs[i,:,:], KX, KV, t[i])
    u0_rs_s[i,:,:] = phase_space_shift(u0_rs, KX, KV, t[i])
    
    # ra = np.absolute(np.fft.fftshift(pdf_kv))
    # #ra = np.real(pdf_kv)
    
    # plt.figure()
    # plt.contourf(XB, VB, pdf_b, cb(pdf_b))
    # plt.colorbar()
    
    # plt.figure()
    # plt.contourf(XB, VB, pdf_b_t, cb(pdf_b_t))
    # plt.colorbar()
    # #plt.contourf(XB, VB, pdf_rs[i,:,:], cb(pdf_rs[i,:,:]))
    
    # plt.figure()
    # plt.contourf(KX*L, KV*(vMax-vMin), ra, cb(ra))
    # plt.show()
    #wvs, amps, angles = b.FFS(den[i,:,:], x, xbar, 1.0/Jx, L, lim, cmplx=True)
    #wvs, amps, angles, aprx = b.FFS(den[i,:,:]-1, x, xbar, 1.0/Jx, L, lim, approximate=True)#cmplx=True)
    #wvs, amps, angles, aprx = b.FFS(pot[i,1:-1,:], x[1:-1,:].flatten(), xbar, 1.0/Jx, L, lim, approximate=True)#cmplx=True)
    # potential spectrum
    #wvs, amps, angles = b.FFS(pot[i,:]-np.mean(pot[i,:]), x, xbar, 1.0/Jx, L, lim, cmplx=True)
    
    # Doppler shift
    phase = np.exp(1j*wvs*group*t[i])
    amps2 = np.multiply(amps, phase)
    #den_ds[i,:] = np.real(np.roll(b.sumFourier(amps2, wvs, xf), n))
    den_ds[i,:] = np.real(b.sumFourier(amps2, wvs, xf))
    #den_ds[i,:] = np.real(np.roll(aprx, n)) # why do I need to roll it?
    
    ax.clear()
    ax.plot(xf, n0.flatten()-np.mean(n0), 'o', label='den')
    ax.plot(xf, den_ds[i,:], 'o--', label='reconstruction')
    plt.axis([xf[-1], xf[0], -1.0e-1, 1.0e-1])
    plt.legend(loc='best')
    plt.pause(0.1)
    
    #wig[i,:,:] = WT_fs(xbar, wvs2/2.0, amps2, L, lim, lim)
    
    #print(np.amax(np.imag(wig)))
    #
    
    #plt.figure()
    #plt.contourf(WX, WK, np.real(wig[i,:,:]), cb(np.real(wig[i,:,:])))
    #plt.colorbar()
    
    #plt.figure()
    #plt.plot(wvs2/k0, np.real(amps2), 'o--', label='real')
    #plt.plot(wvs2/k0, np.imag(amps2), 'o--', label='imag')
    #plt.grid(True)
    #plt.legend(loc='best')
    #plt.show()

# fig1, ax1 = plt.subplots(1,2)
# for i in range(Nt):
#     ax1[0].clear()
#     ax1[1].clear()
    
#     drs = pdf_rs[i,:,:]-u0_rs
#     drs_s = pdf_rs_s[i,:,:]-u0_rs_s[i,:,:]
    
#     ax1[0].contourf(XB, VB, drs, cb(drs))
#     ax1[1].contourf(XB, VB, drs_s, cb(drs_s))
#     plt.pause(1)

### SVD the shifted distribution function
df_s = pdf_rs_s - u0_rs_s
X = df_s.reshape(u0_rs_s.shape[0], u0_rs_s.shape[1]*u0_rs_s.shape[2])

u, s, vh = np.linalg.svd(X, full_matrices=False)

plt.figure()
plt.semilogy(s, 'o')
plt.xlabel('Mode number')
plt.ylabel('Amplitude [arb. units]')


fig, ax = plt.subplots(1,3)
ax[0].contourf(XB, VB, vh[0,:].reshape(XB.shape), cb(vh[0,:]))
ax[1].contourf(XB, VB, vh[1,:].reshape(XB.shape), cb(vh[1,:]))
ax[2].contourf(XB, VB, vh[2,:].reshape(XB.shape), cb(vh[2,:]))
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('First three modes of shifted distribution')

plt.figure()
plt.contourf(XX, TT, den_xf, cb(den_xf))
plt.title('Main thermal frame, $v=0$')
plt.xlabel('Position')
plt.ylabel('Time')
plt.tight_layout()


plt.show()

quit()


plt.figure()
plt.contourf(WX, WK, np.real(wig[-1,:,:]), cb(np.real(wig[-1,:,:])))
plt.colorbar()

X = wig.reshape(Nt, wig.shape[1]*wig.shape[2])

uw, sw, vhw = np.linalg.svd(X)

plt.figure()
plt.plot(sw, 'o')
#plt.show()

fig, ax = plt.subplots()
for i in range(10):
    ax.clear()
    ax.contourf(WX, WK, np.real(vhw[i,:].reshape(wig.shape[1], wig.shape[2])), cb(np.real(vhw[i,:])))
    plt.pause(1)
#plt.colorbar()

plt.show()

quit()



#plt.figure()
#plt.plot(xf, den[idxs[0], :, :].flatten(), linewidth='3', label='t={:0.2f}'.format(t_d[0]))


#### Commented stuff
#print(X.shape)
#print(u.shape)
#print(s.shape)
#print(vh.shape)
#quit()

# vh_rs = vh.reshape((Nt, v.shape[0], v.shape[1])) 

# Y = np.dot(X, vh.T)



# plt.figure()

# plt.xlabel('Velocity')

# plt.figure()
# M = np.amax(avg_pdf[0,:,:])
# plt.plot(flat(v), flat(Y[0,0]*vh_rs[0,:,:] - avg_pdf[0,:,:])/M, 'o--', label='one mode')
# plt.plot(flat(v), flat(Y[0,0]*vh_rs[0,:,:]+Y[0,1]*vh_rs[1,:,:] - avg_pdf[0,:,:])/M, 'o--', label='two modes')
# plt.plot(flat(v), flat(Y[0,0]*vh_rs[0,:,:]+Y[0,1]*vh_rs[1,:,:]+Y[0,2]*vh_rs[2,:,:] - avg_pdf[0,:,:])/M, 'o--', label='three modes')
# plt.ylabel(r'Reconstruction error $\epsilon = (R - f)/|f|_\infty$')
# plt.xlabel('Velocity')
# plt.title('Partial sums of BOT initial distribution function in SVD basis')
# plt.grid(True)
# plt.legend(loc='best')
# plt.tight_layout()

# plt.figure()
# plt.plot(t, Y[:,0], 'o--', linewidth='3', label='Mode 0')
# plt.plot(t, Y[:,1], 'o--', linewidth='3', label='Mode 1')
# plt.plot(t, Y[:,2], 'o--', linewidth='3', label='Mode 2')
# plt.plot(t, Y[:,3], 'o--', linewidth='3', label='Mode 3')
# plt.grid(True)
# plt.xlabel(r'Time $t$')
# plt.ylabel(r'Mode strength')
# plt.legend(loc='best')

# plt.show()
    
# quit()
