from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# Input data : shape 2001 x 79 (frequency x time)

# Load the data

data = []

# Number of datasets with n_max = 1E4
n_data = 10


data = np.zeros((n_data, 79, 2001))

for i in range(n_data):
    temp = np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(i), delimiter = ",")
    data[i] = np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(i), delimiter = ",")

# split
ts = 0.8

