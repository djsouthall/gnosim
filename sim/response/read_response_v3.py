import sys
import numpy as np
import matplotlib.pyplot as plt
import ROOT as root 

# Antenna response
file_ant = np.load("test_ara_antenna_response.npy")

# Electronics response
file_elect = np.load("test_ara_system_response.npy")

freqs, h_fft = np.hsplit(file_ant, 2)
freqs, sys_fft = np.hsplit(file_elect, 2)
h_fft = np.array(np.ravel(h_fft))
sys_fft = np.ravel(sys_fft)

h = np.fft.irfft(h_fft)
sys = np.fft.irfft(sys_fft)

t = [i / (3.0) for i in range(len(h))]

plt.plot(freqs[:len(h_fft)], 10.0 * np.log10(np.absolute(h_fft)))
plt.plot(freqs[:len(h_fft)], 10.0 * np.log10(np.absolute(sys_fft)))
plt.xlabel("Freq. [Hz]")
plt.ylabel("dB")
plt.xlim(0, 1000e6)
plt.show()

plt.plot(t, np.fft.fftshift(h), label="Data")
plt.xlabel("Time [ns]")
plt.ylabel("Voltage")
plt.show()



