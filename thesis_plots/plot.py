import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (14, 6)

n = 36

parameter = np.genfromtxt("/hpcwork/cg457676/data/parameters/parameters_0.csv", delimiter = ",")[n]
strain = np.genfromtxt("/hpcwork/cg457676/data/strains/h_{:05}.csv".format(n), delimiter = ",", dtype = complex)
spec = np.swapaxes(np.genfromtxt("/hpcwork/cg457676/data/spectrograms/spec_{:05}.csv".format(n), delimiter = ","), 0, 1)


np.savetxt("./thesis_plots/chapter_4/parameter.csv", parameter, delimiter = ",")

# Strain plot
fig, ax = plt.subplots()

N = 1000

t = 5 * np.arange(N)
y = strain[: N].real


ax.plot(t, y * 1E25, color = "#e60049")
ax.set_ylabel("Strain $h_+ / 10^{-25}$")
ax.set_xlabel("Time $t / \mathrm{s}$")
ax.set_title("Strain of dataset nr. {:04}".format(n), y = 1.02)

ax.grid()

import matplotlib.colors as colors

plt.savefig("./thesis_plots/chapter_4/strain.png")


# Plot spectrogram
fig, ax = plt.subplots()

x = (np.arange(0, 79) + 0.5) * 2E4 / (3600 * 24)
y = (np.arange(2E3 + 1) + 0.5) * 5E-5
z = spec


pc = ax.pcolormesh(x, y, z, norm = colors.LogNorm(vmin = np.max(spec) * 1E-6, vmax = np.max(spec)))
ax.set_yscale("log")

ax.set_ylim(1E-4, 1E-1)


ax.set_ylabel("Frequency $f$")
ax.set_xlabel("Time $t$ in days")
# ax.colorbar(label=r'Gravitational wave amplitude [1/$\sqrt{\mathrm{Hz}}$]')

plt.colorbar(pc, label=r'Gravitational wave amplitude [1/$\sqrt{\mathrm{Hz}}$]')
ax.set_title("Spectrogram of dataset nr. {:04}".format(n), y = 1.02)


ax.grid(False)

plt.savefig("./thesis_plots/chapter_4/spec.png")
