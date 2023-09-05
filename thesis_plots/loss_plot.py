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
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (10, 5)



run_number = 13


model = keras.models.load_model("./Network/network_output/run_1.{:02}/model_1.{:02}.keras".format(run_number, run_number), compile = False)
history = np.load("./Network/network_output/run_1.{:02}/history_1.{:02}.npy".format(run_number, run_number), allow_pickle='TRUE').item()

epochs = np.arange(1, len(history["loss"]) + 1)

fig, ax = plt.subplots()
ax.plot(epochs, history["loss"], color = "#e60049", label = "training")
ax.plot(epochs, history["val_loss"], color = "royalblue", label = "validation")
ax.axvline(epochs[22], ls = "--", color = "black", lw = 1.5)

ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Squared Error")
ax.set_title("Plot of the loss function", y = 1.02)
ax.legend()
ax.grid()

plt.savefig("./loss.png")