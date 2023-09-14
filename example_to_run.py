"""
For the example we have to run
We have to provide the code with following information
1. value of lambda saved as variable lamb_da

2. m,n are out limit of integration and our domain's end points

3. f(t) is the function in our integral equation

4. u(t) is the function we have to calculate, and we also have to
specify the analytic solution of integral equation

5. k(s,t) is our function inside the integration term for linear IE

6. ku_inside_function(s, t, w, b, v) is our inside term of integral sign
we have to specify it for linear and non-linear integral equation

7. neural_network_final(x, w, b, v) and  sigmoid_(x) this two function will be same for  each example


after providing all the function we have to change only one thing in the nnmodel.py
file and run that code to get the solution

The thing we have to change is only to provide name of this python file to the code


"""
import numpy as np

lamb_da = -1
m = 0
n = 1


def f(t):
    pass
    # return np.cos(t) + 2


def u(t):
    pass
    # return np.cos(t)


def k(s, t):
    pass
    # return 1


def neural_network_final(x, w, b, v):
    z = w*x + b
    no_lin = sigmoid_(z)
    return np.dot(no_lin.T, v)


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def ku_inside_function(s, t, w, b, v):
    pass
    # return 1 + (np.sin(s)) ** 2 + (neural_network_final(s, w, b, v)) ** 2
