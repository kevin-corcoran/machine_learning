import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([[69],[123],[168]])

w = np.linalg.inv(X.T@X)@X.T@y
