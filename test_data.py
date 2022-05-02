import numpy as np

arr = np.random.rand(50, 400, 60)

np.save("./data/arr_1.npy", arr)
