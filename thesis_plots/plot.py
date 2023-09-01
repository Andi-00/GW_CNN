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


parameter = np.genfromtxt("/hpcwork/cg457676/data/parameters/parameters_0.csv", delimiter = ",")


spec = np.swapaxes(np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/" + "pspec0_{:05}.csv".format(s + i), delimiter = ","), 0, 1)

x = (np.arange(0, 79) + 0.5) * 2E4 / (3600 * 24)
y = (np.arange(2E3 + 1) + 0.5) * 5E-5
z = spec


fig, ax = plt.subplots()

pc = ax.pcolormesh(x, y, z)
ax.set_yscale("log")

ax.set_ylim(1E-4, 1E-1)


ax.set_ylabel("Frequency $f$")
ax.set_xlabel("Time $t$ in days")
ax.set_title("{})  log M = {:.2f}, a = {:.2f}".format(s + i, np.log10(parameter[i][0]), parameter[i][2]), y = 1.02)
# ax.colorbar(label=r'Gravitational wave amplitude [1/$\sqrt{\mathrm{Hz}}$]')

plt.colorbar(pc, label=r'Gravitational wave amplitude [1/$\sqrt{\mathrm{Hz}}$]')


ax.grid(False)

plt.savefig("./Network/test_programme/specs/spec_{:01}.png".format(s + i))
