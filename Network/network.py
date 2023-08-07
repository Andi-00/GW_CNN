from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# Input data : shape 79 x 2001 (time x frequency)

# Creating the model of the CNN

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation = "relu", input_shape = (79, 2001)))