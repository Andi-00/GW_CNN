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
n_run = 11

model = keras.models.load_model("./Network/network_output/run_{}/model_{}.keras".format(n_run, n_run), compile = False)

# Generate the test_data
data = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",")
predict = np.ones_like(data)

n_test = 1000

for i in range(0, n_test, 20):
    print(i)
    spec = np.reshape(np.array([np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(9000 + i + j), delimiter = ",") for j in range(20)]), (-1, 79, 2001, 1))

    predict[i : i + 20] = model.predict(spec)

# Difference pred - true
delta = predict - data

np.savetxt("./Network/network_output/run_11/predict.txt", predict, delimiter = ",")
np.savetxt("./Network/network_output/run_11/deltas.txt", delta, delimiter = ",")

delM = 3 * delta[:, 0]
deld = 99 * delta[:, 1]
dela = delta[:, 2]
dele = 0.6 * delta[:, 3]
delp = 6 * delta[:, 4]

# fig, ax = plt.subplots()
# ax.hist(delM, color = "#e60049", edgecolor = "black")

print(min(delM), max(delM), np.mean(delM))
print(min(deld), max(deld), np.mean(deld))
print(min(dela), max(dela), np.mean(dela))
print(min(dele), max(dele), np.mean(dele))
print(min(delp), max(delp), np.mean(delp))

print()
print(np.mean(predict[:, 2]))