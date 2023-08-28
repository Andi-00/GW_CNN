from tensorflow import keras
import tensorflow as tf
import numpy as np

# Number of datasets with n_max = 1E4
n_data = 2

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

train_labels = np.reshape(parameter[: a], (-1, 4, 1))
valid_labels = np.reshape(parameter[a : b], (-1, 4, 1))
test_labels = np.reshape(parameter[b :], (-1, 4, 1))

print("end loading data")
dauer = time.time() - s
print("Dauer : {:02}:{:02} (min:sec)".format(int(dauer // 60), int(dauer % 60)))




# Model LSTM
model0 = keras.models.Sequential()

model0.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
model0.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model0.add(keras.layers.MaxPooling2D((1,2)))

model0.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model0.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model0.add(keras.layers.MaxPooling2D((2,2)))

model0.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model0.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model0.add(keras.layers.MaxPooling2D((2,2)))

model0.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model0.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model0.add(keras.layers.MaxPooling2D((2,2)))


# Prep for recurrent layer
model0.add(keras.layers.Reshape(target_shape = (128, -1)))
model0.add(keras.layers.LSTM(units = 256))

# # Dense Layer

model0.add(keras.layers.Dense(256, activation = 'relu'))
model0.add(keras.layers.Dense(256, activation = 'relu'))
model0.add(keras.layers.Dense(4, activation = "relu"))
# mein Andi ist der beste!! u got this bebi <3




def schedular(epoch, lr):
    if epoch < 10: return lr
    else: return lr * tf.math.exp(-0.1)


def custom_loss(y_true, y_pred):
        val = tf.constant([4.0, 1.0, 0.0, 0.1, 10.0])
        val = y_true + val

        metric = tf.math.abs(y_true - y_pred) / val


        return tf.reduce_mean(metric, axis = -1)

early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = True, verbose = 1)



models = [model0]
run_number = 15

for model in models:

    model.summary()

    model.compile(optimizer = "Adam",
            loss = "mse",
            metrics=[custom_loss])

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedular)

    # generator 
    # history = model.fit(train_generator, validation_data = valid_generator, epochs = 40, verbose = 2)
    # eval = model.evaluate(test_generator)

    history = model.fit(x = train_data, y = train_labels, validation_data = (valid_data, valid_labels), epochs = 100, callbacks = [lr_schedule, early_stopping], verbose = 2)

    # No callbacks
    # history = model.fit(x = train_data, y = train_labels, validation_data = (valid_data, valid_labels), epochs = 200, verbose = 2)

    eval = model.evaluate(x = test_data, y = test_labels)

    print(eval)

    # save the model
    model.save("./network_output/run_1.{:02}/model_1.{:02}.keras".format(run_number, run_number))

    # Save the history
    np.save('./network_output/run_1.{:02}/history_1.{:02}.npy'.format(run_number, run_number), history.history)
    # history = np.load("./my_first_model.keras", allow_pickle='TRUE').item()


    run_number += 1
