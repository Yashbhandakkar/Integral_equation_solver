import numpy as np
lamb_da = -1/3
m = 0
n = 1

def f(t):
    return np.exp(2*t + 1/3)


def u(t):
    return np.exp(2*t)


def k(s, t):
    return np.exp(2 * t - (5/3) * s)


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def neural_network_final(x, w, b, v):
    z = w*x + b
    no_lin = sigmoid_(z)
    return np.dot(no_lin.T, v)


def ku_inside_function(s, t, w, b, v):
    return k(s, t)*neural_network_final(s, w, b, v)
