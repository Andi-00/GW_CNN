from matplotlib import pyplot as plt
import numpy as np

n_data = 10

parameter = np.genfromtxt("/hpcwork/cg457676/data/parameters/parameters_0.csv", delimiter = ",")[: n_data]

for i in range(n_data):
    spec = np.genfromtxt("/hpcwork/cg457676/data/Processed_Data_0/" + "pspec0_{:05}.csv".format(i))
    
