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



# Number of datasets with n_max = 1E4
n_data = 10000

tf.random.set_seed(1234)

# Read CSV files (parameter and data sets)
parameter = np.zeros((n_data, 5))

for i in range((n_data - 1) // 1000 + 1):
    parameter[1000 * i : min(1000 * (i + 1), n_data)] = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), delimiter = ",")[: min(1000, n_data - i * 1000)]


files = ["/hpcwork/cg457676/data/Processed_Data/" + "pspec_{:05}.csv".format(i) for i in range(n_data)]


# Generators for reading the data sets

data_input_shape = (79, 2001, 1)
lab_inut_shape = (5, 1)


# data generator
def data_generator(file_paths, labels, split = "train", batchsize = 20):

    if split == "train" :
        a = 0
        b = 0.8

    elif split == "validation":
        a = 0.8
        b = 0.9
    else :
        a = 0.9
        b = 1
    
    n = len(file_paths)
    file_pth = file_paths[int(a * n) : int(b * n)]
    lab = labels[int(a * n) : int(b * n)]

    for i in range(0, len(file_pth), batchsize):

        data_paths = file_pth[i : i + batchsize]
        data = np.reshape(np.array([np.genfromtxt(path, delimiter = ",") for path in data_paths]), (-1, 79, 2001, 1))
        labs = np.reshape(lab[i : i + batchsize], (-1, 5, 1))

        yield data, labs

output = (tf.TensorSpec(shape = [None, *data_input_shape], dtype = tf.float64), tf.TensorSpec(shape = [None, *lab_inut_shape], dtype = tf.float64))

train_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "train"), output_signature = output)
valid_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "validation"), output_signature = output)
test_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "test"), output_signature = output)


# Creating the model of the CNN

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

# Dense Layers

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(64, activation = 'relu'))
model.add(keras.layers.Dense(5, activation = "relu"))


# Test Locally connected
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,6)))

# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,6)))

# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((1,4)))

# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.LocallyConnected2D(128, (3, 3), activation = "relu"))

# # Dense Layer

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation = 'relu'))
# model.add(keras.layers.Dense(128, activation = 'relu'))
# model.add(keras.layers.Dense(5, activation = "relu"))




model.summary()

def custom_loss(y_true, y_pred):
    val = tf.constant([4.0, 1.0, 0.0, 0.1, 10.0])
    val = y_true + val

    metric = tf.math.abs(y_true - y_pred) / val


    return tf.reduce_mean(metric, axis = -1)


model.compile(optimizer='adam',
              loss = "mse",
              metrics=[custom_loss])


history = model.fit(train_generator, validation_data = valid_generator, epochs = 50, verbose = 2)

eval = model.evaluate(test_generator)

print(eval)

run_number = 14

# save the model
model.save("./network_output/run_{}/model_{}.keras".format(run_number, run_number))

# Save the history
np.save('./network_output/run_{}/history_{}.npy'.format(run_number, run_number), history.history)
# history = np.load("./my_first_model.keras", allow_pickle='TRUE').item()

# fig, ax = plt.subplots()

# ax.plot(history.history["loss"], label = "loss", color = "royalblue")
# ax.plot(history.history["val_loss"], label = "val_loss", color = "crimson")
# ax.legend()
# ax.grid()

# plt.savefig("./run0_loss_plot.png")