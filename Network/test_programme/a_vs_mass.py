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
plt.rcParams['figure.figsize'] = (10, 6)

data = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",")

M = data[:, 1]

delta = np.genfromtxt("./Network/network_output/run_1.13/deltas.txt", delimiter = ",")

a = delta[:, 2]

fig, ax = plt.subplots()

ax.scatter(M, np.abs(a), color = "royalblue")
ax.grid()

ax.set_ylabel(r"$|a_\mathrm{True} - a_\mathrm{Prediction}|$")
ax.set_xlabel(r"$\tilde M_\mathrm{True}$")

plt.savefig("./a_M(run1.13).png")