from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# history = keras.models.load_model("./my_first_model.keras")

history = np.load('./Network/model_history.npy',allow_pickle='TRUE').item()

print(history)

fig, ax = plt.subplots()

ax.plot(history["loss"], label = "loss", color = "royalblue")
ax.plot(history["val_loss"], label = "val_loss", color = "crimson")
ax.legend()
ax.grid()

plt.savefig("./test.png")