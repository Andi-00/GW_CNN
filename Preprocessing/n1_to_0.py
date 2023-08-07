import numpy as np

def trafo(array):
    temp = np.copy(array)
    temp[temp < 0] = 0
    return temp



for i in range(10000):

    data = np.genfromtxt("/hpcwork/cg457676/data/Processed_Data/pspec_{:05}.csv".format(i), delimiter = ",")
    
    d = trafo(data)

    np.savetxt("/hpcwork/cg457676/data/Processed_Data_0/pspec0_{:05}.csv".format(i), d, delimiter = ",")