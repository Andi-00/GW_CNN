import numpy as np
import pandas as pd
import seaborn as sns
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


data = pd.read_csv("/hpcwork/cg457676/data/parameters/parameters_0.csv", delimiter = ",", names = ["Mass", "Distance", "Spin", "Eccentricity", "Seperation"])

for i in range(1, 10):
    d = data = pd.read_csv("/hpcwork/cg457676/data/parameters/parameters_{}.csv".format(i), delimiter = ",", names = ["Mass", "Distance", "Spin", "Eccentricity", "Seperation"])
    data.append(d)

data["Mass"] = np.log10(data["Mass"])

print(np.min(data), np.max(data))

sns.set_theme()

plot = sns.PairGrid(data, height = 2)

plot.map_upper(sns.kdeplot, fill = True)
plot.map_lower(sns.scatterplot, s = 5)
plot.map_diag(sns.histplot, kde=True)


plot.fig.suptitle('Distributions and Correlations of the generated Parameter', y = 1.02)
plt.savefig("./thesis_plots/chapter_4/corr.png")

# plot.fig.suptitle('Correlations in run {}'.format(n_run))
# plt.savefig("./Network/network_output/run_{}/corr.png".format(n_run))