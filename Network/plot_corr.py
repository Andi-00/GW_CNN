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
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (10, 6)


n_run = 13

data = pd.read_csv("./Network/network_output/run_1.{:02}/deltas.txt".format(n_run), delimiter = ",", names = [r"$\Delta_0$", r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$", r"$\Delta_4$"])
# data = pd.read_csv("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",", names = ["Mass", "Distance", "Spin", "Eccentricity", "Seperation"])

data[r"$\Delta_0$"] = 3 * data[r"$\Delta_0$"]
data[r"$\Delta_1$"] = 99 * data[r"$\Delta_1$"]
data[r"$\Delta_2$"] = 1 * data[r"$\Delta_2$"]
data[r"$\Delta_3$"] = 0.6 * data[r"$\Delta_3$"]
data[r"$\Delta_4$"] = 6 * data[r"$\Delta_4$"]

print(np.min(data), np.max(data))

values = [[data["$\Delta_0$"], data["$\Delta_1$"]], [data["$\Delta_0$"], data["$\Delta_2$"]], 
          [data["$\Delta_0$"], data["$\Delta_4$"]], [data["$\Delta_1$"], data["$\Delta_4$"]], 
          [data["$\Delta_2$"], data["$\Delta_4$"]]]

nums = [[0, 1], [0, 2], [0, 4], [1, 4], [2, 4]]
i = 0

for v in values:

    x, y = v[0], v[1]
    
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    print(m, c)

    t = np.linspace(- np.max(np.abs(x)), np.max(np.abs(x)), 100)

    fig, ax = plt.subplots()

    ax.scatter(x, y, color = "#e60049")
    ax.plot(t, m * t + c, color = "black")
    ax.grid()
    plt.savefig("./thesis_plots/plots/chapter_5/{}_{}.png".format(nums[i][0], nums[i][1]))

    i += 1



sns.set_theme(font_scale = 0.9, font = "serif")

plot = sns.PairGrid(data, height = 1.2)

plot.map_upper(sns.kdeplot, fill = True)
plot.map_lower(sns.scatterplot, s = 5)
plot.map_diag(sns.histplot)


plt.savefig("./thesis_plots/plots/Correlations.png".format(n_run))



# plot.fig.suptitle('Correlations in run {}'.format(n_run))
# plt.savefig("./Network/network_output/run_{}/corr.png".format(n_run))