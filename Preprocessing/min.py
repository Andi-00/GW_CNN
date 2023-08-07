import numpy as np

# Code to extract the minima and maxima of the spectrograms

# minima = []
# maxima = []


# for i in range(10000):

#     print(i)

#     temp = np.genfromtxt("/hpcwork/cg457676/data/spectrograms/spec_{:05}.csv".format(i), delimiter = ",")

    

#     minima.append(np.min(temp))
#     maxima.append(np.max(temp))


# np.savetxt("./maxima.csv", maxima, delimiter = ",")
# np.savetxt("./minima.csv", minima, delimiter = ",")


maxi = np.genfromtxt("./maxima.csv", delimiter = ",")
mini = np.genfromtxt("./minima.csv", delimiter = ",")

min0 = np.copy(mini)

# list of minima, where 0 is replaced by 1 to find the true minima > 0
min0[min0 == 0] = 1

print()
print(r"Global Maximum $m_+$" + "= {:.4E}".format(np.max(maxi)))
print(r"Minimum of the Maxima $\tilde m_-$" + "= {:.4E}\n".format(np.min(maxi)))

print(r"Global Minimum $m_-$" + "= {:.4E}".format(np.min(min0)))
print(r"Maximum of the Minima $\tilde m_+$" + "= {:.4E}\n".format(np.max(mini)))

print(r"Proposed $m_-$ = " + "{:.4E}".format(np.min(maxi) * 1E-6))

print("\n{:.4E}".format((min(maxi) * 2E-5) ** 2 / max(maxi)))

# We will take m- = 1E-30 and m+ = 1.156 * 1E-21