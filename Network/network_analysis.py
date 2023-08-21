from tensorflow import keras
import tensorflow as tf
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

# Load the model
n_run = 16

model = keras.models.load_model("./Network/network_output/run_{}/model_{}.keras".format(n_run, n_run), compile = False)

# Generate the test_data
data = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",")
predict = np.ones_like(data)

n_test = 1000

for i in range(0, n_test, 20):
    print(i)
    spec = np.reshape(np.array([np.genfromtxt("/hpcwork/cg457676/data/Processed_Data/pspec_{:05}.csv".format(9000 + i + j), delimiter = ",") for j in range(20)]), (-1, 79, 2001, 1))

    predict[i : i + 20] = model.predict(spec)

# Difference pred - true
delta = predict - data

np.savetxt("./Network/network_output/run_{}/predict.txt".format(n_run), predict, delimiter = ",")
np.savetxt("./Network/network_output/run_{}/deltas.txt".format(n_run), delta, delimiter = ",")

# delta = np.genfromtxt("./Network/network_output/run_11/deltas.txt", delimiter = ",")

delM = 3 * delta[:, 0]
deld = 99 * delta[:, 1]
dela = delta[:, 2]
dele = 0.6 * delta[:, 3]
delp = 6 * delta[:, 4]


# print(min(delM), max(delM), np.mean(delM))
# print(min(deld), max(deld), np.mean(deld))
# print(min(dela), max(dela), np.mean(dela))
# print(min(dele), max(dele), np.mean(dele))
# print(min(delp), max(delp), np.mean(delp))

hists = [delM, deld, dela, dele, delp]
names = ["mass", "dist", "spin", "ecc", "sep"]
axlabs = ["$M_\mathrm{prediction} - M_\mathrm{true}$", "$d_\mathrm{prediction} - d_\mathrm{true}$",
          "$a_\mathrm{prediction} - a_\mathrm{true}$", "$e_\mathrm{prediction} - e_\mathrm{true}$",
          "$p_\mathrm{prediction} - p_\mathrm{true}$"]

for i in range(len(hists)):

    width = max(hists[i]) - min(hists[i])
    max_err = max(np.abs(hists[i]))
    binw = width / 13

    bi = (np.arange(-8, 8) + 0.5) * max_err / 8

    mean = np.mean(hists[i])
    std = np.std(hists[i], ddof = 1)

    print(mean, std)

    fig, ax = plt.subplots()
    ax.hist(hists[i], bins = bi, color = "#e60049", edgecolor = "black")
    ax.axvline(mean, ls = "--", color = "black", label = "$\mu = $" + " {:.3f}".format(mean), zorder = 10)
    ax.axvline(mean + std, ls = ":", color = "black", label = "$\sigma = $" + " {:.3f}".format(std), zorder = 10)
    ax.axvline(mean - std, ls = ":", color = "black", zorder = 10)

    ax.set_ylabel("Number of Counts")
    ax.set_xlabel(axlabs[i])
    ax.set_title("Deviation of the Prediction and Input", y = 1.02)

    ax.legend()

    plt.savefig("./Network/network_output/run_{}/hists/r{}_hist_{}.png".format(n_run, n_run, names[i]))