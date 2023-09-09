import numpy as np
import pandas as pd
# import seaborn as sns
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
plt.rcParams['figure.figsize'] = (8, 5)

title = ["$\log_{10} M / M_\odot$", "$d_L$", "$a$", "$e_0$", "$p_0 / M$"]

# data = pd.read_csv("/hpcwork/cg457676/data/parameters/parameters_0.csv", delimiter = ",", names = title)

# for i in range(1, 10):
#     d = pd.read_csv("/hpcwork/cg457676/data/parameters/parameters_{}.csv".format(i), delimiter = ",", names = title)
#     data.append(d)

# titles = [r"$\Delta_0$", r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$", r"$\Delta_4$"]

# data = pd.read_csv("./Network/network_output/run_1.13/deltas.txt", delimiter = ",", names = titles)

# print(data.corr())

# data.corr().to_csv("./thesis_plots/corr_coef.csv")






# data[title[0]] = np.log10(data[title[0]])

# print(np.min(data), np.max(data))

# sns.set_theme(font_scale = 0.9, font = "serif")

# plot = sns.PairGrid(data, height = 1.2)

# plot.map_upper(sns.kdeplot, fill = True)
# plot.map_lower(sns.scatterplot, s = 5)
# plot.map_diag(sns.histplot, kde=True)


# plt.savefig("./thesis_plots/plots/chapter_4/corr.png")

# plot.fig.suptitle('Correlations in run {}'.format(n_run))
# plt.savefig("./Network/network_output/run_{}/corr.png".format(n_run))


data = np.genfromtxt("./Network/network_output/old_runs/run_12/deltas.txt", delimiter = ",")

h = data[:, 2]

width = max(h) - min(h)
max_err = max(np.abs(h))
binw = width / 13

bi = (np.arange(-8, 8) + 0.5) * max_err / 8

    

mean = np.mean(h)
std = np.std(h, ddof = 1)


    

print(mean, std)

fig, ax = plt.subplots()
(n, bins, patches) = ax.hist(h, bins = bi, color = "#e60049", edgecolor = "black")

z = bi[-1] * 1.2

k = np.linspace(-z, z, 10000)

a = np.max(n)
a = 1000 * (bi[1] - bi[0]) / np.sqrt(2 * np.pi * std ** 2)

# gauss = a * np.exp(-(k - mean) ** 2 / (2 * std ** 2))

# ax.plot(k, gauss, color = "black", label = "Fit")


ax.axvline(mean, ls = "--", color = "black", label = "$\mu = $" + " {:.3f}".format(mean), zorder = 10)
ax.axvline(mean + std, ls = ":", color = "black", label = "$\sigma = $" + " {:.3f}".format(std), zorder = 10)
ax.axvline(mean - std, ls = ":", color = "black", zorder = 10)

ax.set_ylabel("Number of Counts")
ax.set_xlabel(r"$\Delta_2 = a_\mathrm{predict} - a_\mathrm{true}$")
ax.legend()

plt.savefig("thesis_plots/plots/spin_hist.png")