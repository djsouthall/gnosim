import numpy as np
import matplotlib.pyplot as plt

response_freqs = [float(i) * 2.92968750e+06 for i in range(2**14 + 1)]
response_angs = np.arange(0, 190, 10).astype(float)
response_vs_angle = np.array([np.zeros(2**14 + 1) for i in range(len(response_angs))])

for i in range(len(response_vs_angle)):
    for j in range(len(response_freqs)):
        response_vs_angle[i][j] = i * 10.0 / 180.0

np.savez("upsampled_sim_response_calibration", response_angs=response_angs, response_freqs=response_freqs, response_vs_angle=response_vs_angle)

for i in range(len(response_vs_angle)):
    plt.plot(response_freqs, response_vs_angle[i], label=response_angs[i])
plt.legend()
plt.xlabel("Freq. [Hz]")
plt.ylabel("Response [1/m]")
plt.xlim(0, 5e9)
plt.show()
