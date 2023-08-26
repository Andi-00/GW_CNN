import tensorflow as tf
from tensorflow import keras
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



run_number = 12


model = keras.models.load_model("./Network/network_output/run_1.{:02}/model_1.{:02}.keras".format(run_number, run_number), compile = False)
history = np.load("./Network/network_output/run_1.{:02}/history_1.{:02}.npy".format(run_number, run_number), allow_pickle='TRUE').item()

n_test = 1000

data = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",", skip_header = 1000 - n_test)
predict = np.ones_like(data)


for i in range(0, n_test, 20):
    print(i)
    spec = np.reshape(np.array([np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(9000 + i + j), delimiter = ",") for j in range(20)]), (-1, 79, 2001, 1))

    predict[i : i + 20] = model.predict(spec)

dM = np.abs(data[:, 0] - predict[:, 0])
dd = np.abs(data[:, 1] - predict[:, 1])
da = np.abs(data[:, 2] - predict[:, 2])
de = np.abs(data[:, 3] - predict[:, 3])
dp = np.abs(data[:, 4] - predict[:, 4])

Mp = dM / (data[:, 0] + 4)
dp = dd / (data[:, 1] + 1)
ap = da / data[:, 2]
ep = de / (data[:, 1] + 0.1)
pp = dp / (data[:, 1] + 10)

values = np.empty((6, 3), dtype = object)

values[0, 0] = " "
values[0, 1] = "mean dist"
values[0, 2] = "mean percentage error"
values[1, 0] = "log10 Mass"
values[2, 0] = "distance"
values[3, 0] = "spin"
values[4, 0] = "eccentricity"
values[5, 0] = "seperation"

dis = [np.mean(dM), np.mean(dd), np.mean(da), np.mean(de), np.mean(dp)]
per = [np.mean(Mp), np.mean(dp), np.mean(ap), np.mean(ep), np.mean(pp)]

values[1:, 1] = dis
values[1:, 2] = per

print(values)

np.savetxt("./Network/network_output/run_1.{:02}/eval_1.{:02}.csv".format(run_number, run_number), values, delimiter = ",", fmt="%s")

epochs = np.arange(1, len(history["loss"]) + 1)

fig, ax = plt.subplots()
ax.plot(epochs, history["loss"], color = "#e60049", label = "training")
ax.plot(epochs, history["val_loss"], color = "royalblue", label = "validation")

ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Squared Error")
ax.set_title("Evaluation of run_1.{:02}".format(run_number), y = 1.02)
ax.legend()
ax.grid()

plt.savefig("./Network/network_output/run_1.{:02}/loss.png".format(run_number))