from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# Input data : shape 2001 x 79 (frequency x time)

# Load the data

data = []

# Number of datasets with n_max = 1E4
n_data = 1000

# Templates for the data sets and the true values
data = np.zeros((n_data, 79, 2001, 1))
labels = np.zeros((n_data, 5))

# loading the data sets and the labels
for i in range(n_data):
    tempd = np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(i), delimiter = ",")

    j = i % 1000
    templ = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i // 1000), delimiter = ",")[j]

    data[i] = np.reshape(tempd, (79, 2001, 1))
    labels[i] = templ

# split
teSp = int(0.8 * n_data)
vaSp = int(0.1 * n_data)
trSP = int(0.1 * n_data)

# Train / Validation / Test - Data
train_data = data[:teSp]
valid_data = data[teSp : teSp + vaSp]
test_data = data[teSp + vaSp:]

train_labels = labels[:teSp]
valid_labels = labels[teSp : teSp + vaSp]
test_labels = labels[teSp + vaSp:]

# Creating the model of the CNN

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (5, 3), activation = "relu", input_shape = (79, 2001, 1)))
model.add(keras.layers.Conv2D(16, (5, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((1,2)))

model.add(keras.layers.Conv2D(32, (5, 3), activation = "relu"))
model.add(keras.layers.Conv2D(32, (5, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
model.add(keras.layers.Conv2D(16, (3, 3), activation = "relu"))
model.add(keras.layers.MaxPooling2D((2,2)))

# Dense Layers

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(5, activation = "relu"))

model.summary()

print(train_labels)
print(train_data)

model.compile(optimizer='adam',
              loss = "mse",
              metrics=['mse'])

history = model.fit(train_data, train_labels, epochs = 100, 
                    validation_data = (valid_data, valid_labels))

# model.save("./my_first_model.keras")

# np.save('./model_history.npy', history.history)

