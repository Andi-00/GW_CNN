import numpy as np

def trafo(a):
    temp = np.copy(a)

    temp[:, 0] = (np.log10(temp[:, 0]) - 4) / 3
    temp[:, 1] = (temp[:, 1] - 1) / 99
    temp[:, 3] = (temp[:, 3] - 0.1) / 0.6
    temp[:, 4] = (temp[:, 4] - 10) / 6

    return temp

for i in range(10):
    print(i)

    data = np.genfromtxt("/hpcwork/cg457676/data/parameters/parameters_{}.csv".format(i), delimiter = ",")

    temp = trafo(data)

    np.savetxt("/hpcwork/cg457676/data/processed_parameter/pro_par{}.csv".format(i), temp, delimiter = ",")
