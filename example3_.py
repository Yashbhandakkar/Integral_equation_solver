import numpy as np

lamb_da = 1/2
m = 0
n = np.pi/2


def f(t):
    return np.sin(t) - (3.14/8)


def u(t):
    return np.sin(t)


def k(s, t):
    return 1


def neural_network_final(x, w, b, v):
    z = w*x + b
    no_lin = sigmoid_(z)
    return np.dot(no_lin.T, v)


# This is for example3
def ku_inside_function(s, t, w, b, v):
    return k(s, t)*(neural_network_final(s, w, b, v))**2


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))
