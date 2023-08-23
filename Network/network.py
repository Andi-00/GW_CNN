from tensorflow import keras
import tensorflow as tf
import numpy as np

# Number of datasets with n_max = 1E4
n_data = 3000

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

# Generators for reading the data sets

# data_input_shape = (79, 2001, 1)
# lab_inut_shape = (5, 1)


# # data generator
# def data_generator(file_paths, labels, split = "train", batchsize = 32):

#     if split == "train" :
#         a = 0
#         b = 0.8

#     elif split == "validation":
#         a = 0.8
#         b = 0.9
#     else :
#         a = 0.9
#         b = 1
    
#     n = len(file_paths)
#     file_pth = file_paths[int(a * n) : int(b * n)]
#     lab = labels[int(a * n) : int(b * n)]

#     for i in range(0, len(file_pth), batchsize):

#         data_paths = file_pth[i : i + batchsize]
#         data = np.reshape(np.array([np.genfromtxt(path, delimiter = ",") for path in data_paths]), (-1, 79, 2001, 1))
#         labs = np.reshape(lab[i : i + batchsize], (-1, 5, 1))

#         yield data, labs

# output = (tf.TensorSpec(shape = [None, *data_input_shape], dtype = tf.float64), tf.TensorSpec(shape = [None, *lab_inut_shape], dtype = tf.float64))

# train_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "train"), output_signature = output)
# valid_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "validation"), output_signature = output)
# test_generator = tf.data.Dataset.from_generator(lambda : data_generator(files, parameter, split = "test"), output_signature = output)


# Creating the model of the CNN

# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# # Dense Layers

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(64, activation = 'relu'))
# model.add(keras.layers.Dense(64, activation = 'relu'))
# model.add(keras.layers.Dense(5, activation = "relu"))


# Gutes Network (run 19 und co)
# model = keras.models.Sequential()
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
# model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
# model.add(keras.layers.MaxPooling2D((2,2)))

# # # Dense Layer

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(128, activation = 'relu'))
# model.add(keras.layers.Dense(128, activation = 'relu'))
# model.add(keras.layers.Dense(5, activation = "relu"))


# Model 4
model_3 = keras.models.Sequential()
model_3.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (79, 2001, 1)))
model_3.add(keras.layers.Conv2D(32, (3, 3), activation = "relu"))
model_3.add(keras.layers.MaxPooling2D((2,2)))

model_3.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model_3.add(keras.layers.Conv2D(64, (3, 3), activation = "relu"))
model_3.add(keras.layers.MaxPooling2D((2,2)))

model_3.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model_3.add(keras.layers.Conv2D(128, (3, 3), activation = "relu"))
model_3.add(keras.layers.MaxPooling2D((2,2)))

model_3.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model_3.add(keras.layers.Conv2D(256, (3, 3), activation = "relu"))
model_3.add(keras.layers.MaxPooling2D((2,2)))

# # Dense Layer

model_3.add(keras.layers.Flatten())

model_3.add(keras.layers.LSTM(256))

model_3.add(keras.layers.Dense(256, activation = 'relu'))
model_3.add(keras.layers.Dense(256, activation = 'relu'))
model_3.add(keras.layers.Dense(5, activation = "relu"))


# Model 5
model_5 = keras.models.Sequential()
model_5.add(keras.layers.Conv2D(64, (7, 19), activation = "relu", input_shape = (79, 2001, 1)))
model_5.add(keras.layers.MaxPooling2D((1,2)))

model_5.add(keras.layers.Conv2D(128, (5, 15), activation = "relu"))
model_5.add(keras.layers.MaxPooling2D((2,2)))

model_5.add(keras.layers.Conv2D(256, (5, 11), activation = "relu"))
model_5.add(keras.layers.MaxPooling2D((2,2)))

model_5.add(keras.layers.Conv2D(512, (3, 7), activation = "relu"))
model_5.add(keras.layers.MaxPooling2D((2,2)))

# # Dense Layer

model_5.add(keras.layers.Flatten())

model_5.add(keras.layers.Dense(512, activation = 'relu'))
model_5.add(keras.layers.Dense(512, activation = 'relu'))
model_5.add(keras.layers.Dense(5, activation = "relu"))




def schedular(epoch, lr):
    if epoch < 10: return lr
    else: return lr * tf.math.exp(-0.1)


def custom_loss(y_true, y_pred):
        val = tf.constant([4.0, 1.0, 0.0, 0.1, 10.0])
        val = y_true + val

        metric = tf.math.abs(y_true - y_pred) / val


        return tf.reduce_mean(metric, axis = -1)




models = [model_3, model_5]
run_number = 4

for model in models:

    model.summary()

    model.compile(optimizer = "Adam",
            loss = "mse",
            metrics=[custom_loss])

    callback = tf.keras.callbacks.LearningRateScheduler(schedular)

    # generator 
    # history = model.fit(train_generator, validation_data = valid_generator, epochs = 40, verbose = 2)
    # eval = model.evaluate(test_generator)

    history = model.fit(x = train_data, y = train_labels, validation_data = (valid_data, valid_labels), epochs = 50, callbacks = [callback], verbose = 2)
    eval = model.evaluate(x = test_data, y = test_labels)

    print(eval)

    # save the model
    model.save("./network_output/run_1.{:02}/model_1.{:02}.keras".format(run_number, run_number))

    # Save the history
    np.save('./network_output/run_1.{:02}/history_1.{:02}.npy'.format(run_number, run_number), history.history)
    # history = np.load("./my_first_model.keras", allow_pickle='TRUE').item()

    

    run_number += 1
