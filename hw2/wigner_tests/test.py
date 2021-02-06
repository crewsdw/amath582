import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import cmath

### Compare Wigner transforms of interpolant and continuous functions
### Discrete space
# Element edges
Lmax = np.pi
Lmin = -np.pi
x = np.linspace(Lmin, Lmax, num=300)
L = x[-1]-x[0]
J = (x[1]-x[0])/2.0 # jacobian
# Midpoints
xbar = np.array([(x[i+1] + x[i])/2.0 for i in range(x.shape[0]-1)])
# Interpolant on midpoints
### Syntheic data
### Elliptic cosine data
k = 1.0-1.0e-10
m = k**2.0
arg = 4.0*sp.ellipk(m)
y = sp.ellipj(arg*xbar, m)[1]
#y = np.sin(xbar) +

y = xbar**2.0#xbar#np.sin(xbar) + np.sin(3.0*xbar) + np.sin(5.0*xbar) + np.sin(7.0*xbar) + np.sin(9.0*xbar)

# Fourier series coefficients
k0 = 2.0*np.pi/L
wvs = k0*np.arange(x.shape[0])

# Fourier series of piecewise interpolant
def FS(y, xbar, wvs, J, L):
    amps = np.zeros_like(wvs) + 0j
    for i in range(wvs.shape[0]):
        amps[i] = sum(2.0*cmath.exp(-1.0j*wvs[i]*xbar[j])*y[j]*np.sinc(wvs[i]*J / np.pi) for j in range(xbar.shape[0]))
    return (J/L)*amps

# Wigner function of piecewise interpolant
def WT(x, k, c, L):
    # Fund. freq.
    k1 = 2.0*np.pi/L
    # Conjugate coefficients
    cc = np.conj(c)
    # Compute wigner function (slow for debug)
    w = np.zeros((x.shape[0], k.shape[0])) + 0j
    w[:,0] = c[0]*cc[0]*2.0/k1
    for l in range(1,k.shape[0]):
        # Set weights (like "neumann factors")
        weights = np.ones(l)
        weights[0] = 0.5
        weights[-1] = 0.5
        # Sum parts of series
        for n in range(l):
            # First fourier factor
            f1 = c[n]*np.exp(1.0j*k[n]*x)
            # Second fourier factor
            f2 = c[l-n]*np.exp(1.0j*k[l-n]*x)
            # Term in wigner series
            w[:,l] += 4.0*weights[n]*np.multiply(np.conj(f1),f2)*2.0/k1
    # Return wigner function
    return w

### Coefficients
c = FS(y, xbar, wvs, J, L)



### meshgrid
# wavenumber grid
res = 50
k = k0*(np.arange(35))/2.0

X,K = np.meshgrid(xbar, k)

# Slow loop
w = WT(xbar, k, c, L).transpose()

# Real and imaginary parts
wi = np.imag(w)
Mi = np.amax(wi)
wr = np.real(w)
Mr = np.amax(wr)
# Amplitude (log(|\psi| + 1))
wa = np.log(np.absolute(w)+1.0)
Ma = np.amax(wa)

cbr = np.linspace(np.amin(wr/Mr), np.amax(wr/Mr), num=50)
cbi = np.linspace(np.amin(wi/Mi), np.amax(wi/Mi), num=50)
cba = np.linspace(np.amin(wa/Ma), np.amax(wa/Ma), num=50)



plt.show()

###### Compare to Gabor transform
pts = 40 # how many points in the window
windowsize = pts*(xbar[1]-xbar[0]) # scaled windowsize
freqs = np.fft.fftfreq(xbar.shape[0], d=xbar[1]-xbar[0]) # frequencies in fft IN HERTZ
gabor_sg = np.zeros((xbar.shape[0], xbar.shape[0])) + 0j # gabor spectrogram
fig, ax = plt.subplots()
size = int(xbar.shape[0]//2)
for i in range(xbar.shape[0]):
    # Build window
    r = (np.arange(2*pts)-pts) + size
    window = np.zeros_like(xbar)
    window[r] = np.exp(1.0)*np.exp(-1.0/(1.0 - xbar[r]**2.0/windowsize**2.0+1.5e-10))
    # Shift window
    window = np.roll(window, i-size)
    #window[rp] = np.exp(-1.0/(1.0 - shift_p[rp]**2.0/windowsize**2.0)+1.0e-16)
    weighted_y = np.multiply(window, y)
    gabor_sg[i,:] = np.fft.fft(weighted_y)
    ax.plot(xbar, weighted_y)
    #ax.plot(freqs, np.absolute(gabor_sg[i,:]), 'o--')
    ax.set_ylim([np.amin(y), np.amax(y)])
    plt.pause(0.05)
    ax.clear()
    #plt.figure()
    #plt.plot(shift_p)
    #plt.show()
    #print(gabor_fft.shape)
    #quit()

XX, KK = np.meshgrid(xbar, np.fft.fftshift(freqs), indexing='ij')

ga = np.log(np.absolute(np.fft.fftshift(gabor_sg))+1.0)
Mg = np.amax(ga)

cbg = np.linspace(np.amin(ga/Mg), np.amax(ga/Mg), num=100)

### Plot Wigner function
plt.figure()
plt.contourf(X, K/k0, wr/Mr, cbr)
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Mode number l')
plt.title(r'Wigner-Ville Distribution Re$(\psi)$')
plt.tight_layout()

plt.figure()
plt.contourf(X, K/k0, wi/Mi, cbi)
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Mode number l')
plt.title(r'Wigner-Ville Distribution Im$(\psi)$')
plt.tight_layout()

plt.figure()
plt.contourf(X, K/k0, wa/Ma, cba)
plt.colorbar()
plt.xlabel('Position x')
plt.ylabel('Mode number n')
plt.title(r'Wigner-Ville Distribution ln$(|\psi|+1)$')
plt.tight_layout()

### Plot Gabor transform
plt.figure()
plt.contourf(XX, L*KK, ga/Mg, cbg)#, extend='both')
plt.colorbar()
plt.title('Gabor transform with bump-function window')
plt.xlabel('Position x')
plt.ylabel('Mode number n')
plt.tight_layout()

### Function and fourier coefficients
plt.figure()
plt.plot(xbar, y, 'o')
plt.xlabel('Position x')
plt.ylabel('Funky wave y(x)')
plt.grid(True)
plt.tight_layout()

plt.figure()
plt.plot(wvs/k0, np.real(c), 'o')
plt.plot(wvs/k0, np.imag(c), 'o')

plt.show()

