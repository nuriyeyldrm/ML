import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial


def LagrangeInterpolation(a, b, n, s):
    rand = np.random.uniform(a, b, n)
    x_train = np.sort(rand)
    y_train = np.sin(x_train)

    epsilon = np.random.normal(0, s, n)

    lagrange_result = lagrange(x_train + epsilon, y_train)
    x_test = np.random.uniform(a, b, n)
    y_test = np.sin(x_test)

    x_train_w_lagrange = Polynomial(lagrange_result.coef[::-1])(x_train)
    x_test_w_lagrange = Polynomial(lagrange_result.coef[::-1])(x_test)

    train_error = np.mean((x_train_w_lagrange - y_train) ** 2)
    test_error = np.mean((x_test_w_lagrange - y_test) ** 2)
    return train_error, test_error


a = -3.14
b = 3.14
n = [10, 20, 30, 50, 100]
s = 0
print("without noise: \n")

for i in n:
    print("a = {}, b = {}, n = {}".format(a, b, i))
    e1 = LagrangeInterpolation(a, b, i, s)
    print("Train Error: ", e1[0], "Test Error: ", e1[1], "\n")


a = -3.14
b = 3.14
n = 18
s = [0, 1, 2, 3, 5]
print("With noise\n")

for i in s:
    print("a = {}, b = {}, n = {}, noise ratio = {}".format(a, b, n, i))
    e2 = LagrangeInterpolation(a, b, n, i)
    print("Train Error: ", e2[0], "Test Error: ", e2[1], "\n")
