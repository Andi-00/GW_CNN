import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt


n_data = 10

# Read CSV files from List
parameter = np.zeros((n_data, 5))

for i in range((n_data - 1) // 1000 + 1):
    parameter[1000 * i : min(1000 * (i + 1), n_data)] = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), delimiter = ",")[: min(1000, n_data - i * 1000)]



# Dataset with all the indices
indices = tf.data.Dataset.range(n_data)
# indices = tf.data.Dataset.from_tensor_slices(["/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(i) for i in range(n_data)])


def load_data(index):
    return index



dataset = indices.map(load_data)











# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(2, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(5, activation = "relu"))

# dataset = tf.data.Dataset.from_tensor_slices((dataset, parameter[:10]))

# model.summary()

# model.compile(optimizer='adam',
#               loss = "mean_absolute_percentage_error",
#               metrics=['mean_absolute_percentage_error'])

# history = model.fit(x = dataset, epochs = 20, verbose = 2)