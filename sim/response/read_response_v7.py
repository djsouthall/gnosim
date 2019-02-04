import sys
import numpy as np
import matplotlib.pyplot as plt

# Antenna response
file_ant = np.load("ara_antenna_response.npy")

# Electronics response
file_elect = np.load("ara_system_response.npy")

freqs, h_fft = np.hsplit(file_ant, 2)
freqs, sys_fft = np.hsplit(file_elect, 2)
h_fft = np.ravel(h_fft)
sys_fft = np.ravel(sys_fft)
freqs = np.ravel(freqs)

# Somehow, they got imaginary... This fixes it
freqs = freqs.astype(float)

sys_gain = np.absolute(sys_fft)**2
sys_gain *= 2.0 # Because the rfft is missing half the power



h_gain = np.absolute(h_fft)
h_gain *= (1.74 / 3e8) * np.array(freqs)
h_gain *= h_gain 
h_gain *= 2.0 # Because the rfft is missing half the power
h_gain *= 4.0 * np.pi

plt.plot(freqs, 10.0 * np.log10(h_gain), label="Antenna")
plt.plot(freqs, 10.0 * np.log10(sys_gain), label="Elect.")
plt.xlabel("Freq. [Hz]")
plt.ylabel("Realized Gain [dBi]")
plt.legend()
plt.xlim(0, 1000e6)
plt.ylim(-15.0, 80.0)
plt.minorticks_on()
plt.grid(which="both")
plt.show()

h = np.fft.irfft(h_fft)
sys = np.fft.irfft(sys_fft)

t = [i / 1.5 for i in range(len(h))]

plt.plot(t, np.fft.fftshift(h))
plt.xlabel("Time [ns]")
plt.ylabel("Signal [m]") # For h, its actually in meters, unless the fft gives it trouble
plt.show()

convo_fft = h_fft * sys_fft
time = np.fft.irfft(convo_fft)
plt.plot(range(len(time)), time)
plt.show()
