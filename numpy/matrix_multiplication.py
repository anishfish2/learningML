import numpy as np

# Dot Product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.dot(a, b)
print(c)

a = np.array([[5, 7, 10], [2, 4, 8]])
b = np.array([[1, 1], [3, 4], [5, 6]])
c = np.dot(a, b)
print(c)

# Matrix Multiplication
a = np.array([[5, 7, 10], [2, 4, 8]])
b = np.array([[1, 1], [3, 4], [5, 6]])
c = np.matmul(a, b)
print(c)


