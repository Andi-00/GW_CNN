from tensorflow import keras
import tensorflow as tf
import numpy as np

# Number of datasets with n_max = 1E4
n_data = 10000

tf.random.set_seed(1234)

# Read CSV files (parameter and data sets)
parameter = np.zeros((n_data, 5))

for i in range((n_data - 1) // 1000 + 1):
    parameter[1000 * i : min(1000 * (i + 1), n_data)] = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), delimiter = ",")[: min(1000, n_data - i * 1000)]


files = ["/hpcwork/cg457676/data/Processed_Data_0/" + "pspec0_{:05}.csv".format(i) for i in range(n_data)]

a = int(n_data * 0.8)
b = int(n_data * 0.9)

import time

s = time.time()

print("start loading data")

train_data = np.reshape(np.array([np.genfromtxt(f, delimiter = ",") for f in files[: a]]), (-1, 79, 2001, 1))
valid_data = np.reshape(np.array([np.genfromtxt(f, delimiter = ",") for f in files[a : b]]), (-1, 79, 2001, 1))
test_data = np.reshape(np.array([np.genfromtxt(f, delimiter = ",") for f in files[b :]]), (-1, 79, 2001, 1))

print(train_data.shape)

train_labels = np.reshape(parameter[: a], (-1, 5, 1))
valid_labels = np.reshape(parameter[a : b], (-1, 5, 1))
test_labels = np.reshape(parameter[b :], (-1, 5, 1))

print("end loading data")
dauer = time.time() - s
print("Dauer : {:02}:{:02} (min:sec)".format(int(dauer // 60), int(dauer % 60)))




# Model LSTM
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((1,2)))

# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))


# # Prep for recurrent layer
# model.add(keras.layers.Reshape(target_shape = (64, -1)))
# model.add(keras.layers.LSTM(units = 256))

# # # Dense Layer

# model.add(keras.layers.Dense(256, activation = 'relu'))
# model.add(keras.layers.Dense(256, activation = 'relu'))
# model.add(keras.layers.Dense(5, activation = "relu"))
# mein Andi ist der beste!! u got this bebi <3

# Make the data of shape (-1, 80, 2001, 1)

# train_data = np.pad(train_data, ((0, 0), (0, 1), (0, 0), (0, 0)))[:, :, : 2000, :]
# train_data = np.reshape(train_data, (-1, 25, 80, 80, 1))

# valid_data = np.pad(valid_data, ((0, 0), (0, 1), (0, 0), (0, 0)))[:, :, : 2000, :]
# valid_data = np.reshape(valid_data, (-1, 25, 80, 80, 1))

# test_data = np.pad(test_data, ((0, 0), (0, 1), (0, 0), (0, 0)))[:, :, : 2000, :]
# test_data = np.reshape(test_data, (-1, 25, 80, 80, 1))

# print(train_data.shape)

# LSTM 
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), input_shape = (79, 2001, 1), activation = "relu"))
model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((1,2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Reshape(target_shape = (128, -1)))
model.add(keras.layers.GRU(units = 256))

model.add(keras.layers.Dense(256, activation = 'relu'))
model.add(keras.layers.Dense(256, activation = 'relu'))
model.add(keras.layers.Dense(5, activation = "relu"))




def schedular(epoch, lr):
    if epoch < 10: return lr
    else: return lr * tf.math.exp(-0.02)


def custom_loss(y_true, y_pred):
        val = tf.constant([4.0, 1.0, 0.0, 0.1, 10.0])
        val = y_true + val

        metric = tf.math.abs(y_true - y_pred) / val


        return tf.reduce_mean(metric, axis = -1)

early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 100, restore_best_weights = True, verbose = 1)




run_number = 17


model.summary()

model.compile(optimizer = "Adam",
        loss = "mse",
        metrics=[custom_loss])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedular)

# generator 
# history = model.fit(train_generator, validation_data = valid_generator, epochs = 40, verbose = 2)
# eval = model.evaluate(test_generator)

history = model.fit(x = train_data, y = train_labels, validation_data = (valid_data, valid_labels), epochs = 1000, callbacks = [lr_schedule, early_stopping], verbose = 2)

# No callbacks
# history = model.fit(x = train_data, y = train_labels, validation_data = (valid_data, valid_labels), epochs = 200, verbose = 2)

eval = model.evaluate(x = test_data, y = test_labels)

print(eval)

# save the model
model.save("./network_output/run_1.{:02}/model_1.{:02}.keras".format(run_number, run_number))

# Save the history
np.save('./network_output/run_1.{:02}/history_1.{:02}.npy'.format(run_number, run_number), history.history)
# history = np.load("./my_first_model.keras", allow_pickle='TRUE').item()

