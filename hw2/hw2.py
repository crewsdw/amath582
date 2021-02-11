import numpy as np
import soundfile as sf

import matplotlib.pyplot as plt
import matplotlib as mpl

# My functions
import fourier as f

# Matplotlib Chunk size (for huge number of points)
mpl.rcParams['agg.path.chunksize'] = 10000

### Read file (samplerate in Hz)
data, samplerate = sf.read('music_files/Floyd.wav')#sf.read('music_files/GNR.wav')
# Create time axis (suppose samples are given in midpoints of intervals of length dt)
dt = 1.0/samplerate
Nt = data.shape[0]
t = np.linspace(0, dt*(Nt+1), num=Nt+1)
# Midpoints
tbar = np.array([(t[i+1] + t[i])/2.0 for i in range(t.shape[0]-1)])
# Jacobian
J = dt/2.0
# Length
L = t[-1]-t[0]

### Wavenumbers
k1 = 2.0*np.pi/L
lim0 = 0#15000
lim1 = 80000
wvs0 = k1*np.arange(lim0, lim1)

### Compute fourier coefficients
# Use real for analytic signal trick
c = np.fft.rfft(data)[lim0:lim1]
wvs = np.fft.rfftfreq(data.shape[0], dt)[lim0:lim1]*2.0*np.pi#*(data.shape[0]*dt)
# Actual Fourier coefficients
c = c/(data.shape[0])

res_wig = lim1
wvs_wig = (np.pi/L)*np.arange(res_wig)


### Check out spectrum
plt.figure()
plt.plot(wvs/(2.0*np.pi), np.absolute(c))
plt.show()

### Compute Wigner transform
# Downsample time
t_final = 25.0#tbar[-1]
t_ds = np.linspace(0, t_final, num=200)
# Wigner dist.
wig = f.WT_as_f(t_ds, wvs_wig, c, L).transpose()

### Compare to Gabor transform
gab_lim0 = 15000#15000#int(lim0//2)
gab_lim1 = 80000#int(lim1//2)
pts = 5000 # how many points in the window
windowsize = pts*(tbar[1]-tbar[0]) # scaled windowsize
#freqs = np.fft.fftfreq(xbar.shape[0], d=xbar[1]-xbar[0]) # frequencies in fft IN HERTZ
gabor_sg = np.zeros((t_ds.shape[0], gab_lim1-gab_lim0)) + 0j # gabor spectrogram
#fig, ax = plt.subplots()
size = int(tbar.shape[0]//2)
ratio = int((t_ds[1]-t_ds[0])//(tbar[1]-tbar[0]))
for i in range(t_ds.shape[0]):
    # Build window
    r = (np.arange(2*pts)-pts) + size
    window = np.zeros_like(tbar)
    window[r] = np.exp(1.0)*np.exp(-1.0/(1.0 - tbar[r]**2.0/windowsize**2.0+1.5e-10))
    # Shift window
    window = np.roll(window, ratio*i - size)
    #window[rp] = np.exp(-1.0/(1.0 - shift_p[rp]**2.0/windowsize**2.0)+1.0e-16)
    weighted_y = np.multiply(window, data)
    gab = np.fft.rfft(weighted_y)[gab_lim0:gab_lim1]/(data.shape[0])
    #flt = np.exp(-0.5*(wvs[:gab_lim]/(2.0*np.pi) - 125.0)**2.0/(25.0**2.0))
    flt = 1.0
    gabor_sg[i,:] = np.multiply(gab, flt)
    #Plot to check
    #ax.plot(tbar, weighted_y, 'o')
    #ax.plot(freqs, np.absolute(gabor_sg[i,:]), 'o--')
    #ax.set_ylim([np.amin(data), np.amax(data)])
    #plt.pause(0.001)
    #ax.clear()

# Meshgrid
T, W = np.meshgrid(t_ds, wvs_wig/(2.0*np.pi))
TG, WG = np.meshgrid(t_ds, wvs[gab_lim0:gab_lim1]/(2.0*np.pi))

#### Spectrograms
ga = np.log(np.absolute(gabor_sg).transpose() + 1.0)
#ga = np.absolute(gabor_sg).transpose()
Mg = np.amax(ga)
cbg = np.linspace(np.amin(ga)/Mg, np.amax(ga)/Mg, num=75)

### Gabor
plt.figure()
plt.contourf(TG, WG, ga/Mg, cbg, cmap='Greys')
plt.xlabel(r'Time $t$ [s]')
plt.ylabel(r'Frequency $\omega$ [Hz]')
plt.colorbar()
plt.title(r'Gabor, log$(|\psi| + 1)$')

plt.show()

### Wigner
# Absolute
wa = np.absolute(wig)
Ma = np.amax(wa)
# Real
wr = np.real(wig)
#Mr = np.amax(wr)
cbr = np.linspace(np.amin(wr)/Ma, np.amax(wr)/Ma, num=75)
# Imag
wi = np.imag(wig)
cbi = np.linspace(np.amin(wi)/Ma, np.amax(wi)/Ma, num=75)

plt.figure()
plt.contourf(T, W, wr/Ma, cbr, cmap='Greys')
plt.xlabel(r'Time $t$ [s]')
plt.ylabel(r'Frequency $\omega$ [Hz]')
plt.colorbar()
plt.title('Real part')

### Gabor-Wigner
gw = np.multiply(np.absolute(wig[::2,:]), np.absolute(gabor_sg.transpose()))

gwa = np.log(np.absolute(gw) + 1.0)
#gwa = np.absolute(gw)
Mgw = np.amax(gwa)
cbgw = np.linspace(0.05, 1.0, num=75)

plt.figure()
plt.contourf(TG, WG, gwa/Mgw, cbgw, cmap='Greys', extend='both')
plt.xlabel(r'Time $t$ [s]')
plt.ylabel(r'Frequency $\omega$ [Hz]')
plt.title('Combined distribution log($|\psi_{GW}| + 1$)')
plt.colorbar()

plt.show()
