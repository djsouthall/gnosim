import numpy as np
import matplotlib.pyplot as plt
import copy
import ROOT as root

# Convert ara bicone anntenna simulated response from a ROOT file to npz file that detector.py is looking for

f = root.TFile("./ara_responses.root")

freq_vs_ang = []
freq = []

for i in range(0, 95, 10):
    h1 = root.TGraph()
    h1 = f.Get("cormag_"+str(i))
    xs = np.ndarray(h1.GetN(), 'd', h1.GetX())
    ys = np.ndarray(h1.GetN(), 'd', h1.GetY())
    freq = xs
    ys = np.sqrt(np.power(10.0, np.array(ys) / 10) / (4.0 * np.pi)) # Given in gain, return to response
    freq_vs_ang += [ys]
for i in range(80, -10, -10):
    h1 = root.TGraph()
    h1 = f.Get("cormag_"+str(i))
    xs = np.ndarray(h1.GetN(), 'd', h1.GetX())
    ys = np.ndarray(h1.GetN(), 'd', h1.GetY())
    freq = xs
    ys = np.sqrt(np.power(10.0, np.array(ys) / 10) / (4.0 * np.pi)) # Given in gain, return to response
    freq_vs_ang += [ys]

response_vs_angle = np.array(freq_vs_ang)
response_freqs = np.array(freq)
response_angs = np.arange(0, 90, 5).astype(float)

desired_freqs = [float(i) * 2.92968750e+06 for i in range(2**14 + 1)]
og_response_vs_angle = copy.deepcopy(response_vs_angle)

response_freqs *= 1e9

# Upsample via interpolation
temp = []
for line in response_vs_angle:
    temp += [np.interp(desired_freqs, response_freqs, line, left=0.0, right=0.0)]
response_vs_angle = temp

np.savez("upsampled_sim_response_ara", response_angs=response_angs, desired_freqs=desired_freqs, response_vs_angle=response_vs_angle)

for i in range(len(response_vs_angle)):
    plt.scatter(response_freqs, og_response_vs_angle[i], color='g')
    plt.plot(desired_freqs, response_vs_angle[i], color='b')

plt.xlim(0, 1e9)
plt.show()
