import numpy as np

lamb_da = -1
m = 0
n = 1


def f(t):
    return np.cos(t) + 2


def u(t):
    return np.cos(t)


def k(s, t):
    return 1


def neural_network_final(x, w, b, v):
    z = w*x + b
    no_lin = sigmoid_(z)
    return np.dot(no_lin.T, v)


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def ku_inside_function(s, t, w, b, v):
    return 1 + (np.sin(s))**2 + (neural_network_final(s, w, b, v))**2