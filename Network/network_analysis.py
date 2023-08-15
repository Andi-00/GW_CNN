from tensorflow import keras
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

n_run = 11

model = keras.models.load_model("./Network/network_output/run_{}/model_{}.keras".format(n_run, n_run), compile = False)

data = np.genfromtxt("/hpcwork/cg457676/data/processed_parameter/pro_par9.csv", delimiter = ",")
predict = np.ones_like(data)


for i in range(0, len(data), 20):
    print(i)
    spec = np.reshape(np.array([np.genfromtxt("/hpcwork/cg457676/data/Processed_Data/pspec_{:05}.csv".format(9900 + j), delimiter = ",") for j in range(20)]), (-1, 79, 2001, 1))

    predict[i : i + 20] = model.predict(spec)

