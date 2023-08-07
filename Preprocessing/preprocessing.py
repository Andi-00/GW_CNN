import numpy as np
from matplotlib import pyplot as plt

m0 = 1E-30
m1 = 1.156 * 1E-21

def trafo(array):
    temp = np.copy(array)
    temp[temp < m0] = m0 ** 2 / m1

    temp = np.log10(temp / m0) / np.log10(m1 / m0)

    return temp



for i in range(10000):

    data = np.genfromtxt("/hpcwork/cg457676/data/spectrograms/spec_{:05}.csv".format(i), delimiter = ",")
    
    d = trafo(data)

    print("{:05}: Max = {:.4E}".format(i, np.max(d)))    

    np.savetxt("/hpcwork/cg457676/data/Processed_Data/pspec_{:05}.csv".format(i), d, delimiter = ",")