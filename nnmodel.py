"""
This is our main code
we only have to change the name of python file in below  line that's it
for example
from example2_ import *
from example3_ import *
"""

from example_to_run import *
# k, f, lamb_da, m, n, u,sigmoid_,ku_inside_function
# we will get everything


from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

H = 10  # number of hidden neurons


def _compute_error(p):
    """Compute the error function using the current parameter values."""

    # Unpack the network parameters (hsplit() returns views, so no copies are made
    (w, b, v) = np.hsplit(p, 3)
    x = np.arange(m, n, 0.1)
    length = len(x)

    # Compute the forward pass through the network.
    z = np.outer(x, w) + b
    s = sigmoid_(z)
    N = s.dot(v)

    integration_ = np.zeros(length)
    for i in range(0, length):
        kk = x[i]
        t = float(kk)
        integration_[i], err = quad(ku_inside_function, m, n, args=(t, w, b, v))
    second_part_ie = lamb_da * integration_

    g = N - (f(x) + second_part_ie)
    l_ = np.sum(g ** 2)
    return l_


def train_minimize(train_alg):
    """Train the network using the SciPy minimize() function. """

    # Assemble the network parameters into a single 1-D vector for
    # use by the minimize() method.
    w = np.random.rand(10)
    b = np.random.rand(10)
    v = np.random.rand(10)
    print("Initial Value of weights")
    print("---------")
    print("w", w)
    print("b", b)
    print("v", v)
    print("---------")
    p = np.hstack((w, b, v))
    # Minimize the error function to get the new parameter values.
    res = minimize(_compute_error, p, method=train_alg)
    # Unpack the optimized network parameters.
    opt_w = res.x[0:H]
    opt_b = res.x[H: 2 * H]
    opt_v = res.x[2 * H: 3 * H]
    return opt_w, opt_b, opt_v


# Neural Network output after we get our parameters
train_algo = 'BFGS'
x = np.arange(0, 1.01, 0.1)
w_, b_, v_ = train_minimize(train_algo)
print("Initial Value of weights")
print("---------")
print("w", w_)
print("b", b_)
print("v", v_)
print("---------")
z = np.outer(x, w_) + b_
s = sigmoid_(z)
N = s.dot(v_)

# We are plotting the result below this
plt.plot(x, N, label='$NN_ solution$')
plt.plot(x, u(x), label='$Analytic solution$')
plt.xlabel('$x$')
plt.ylabel('solution')
plt.grid()
plt.legend()
plt.title("solution by NN and analytic method")
plt.show()

# error between actual solution and NN solution
plt.plot(x, N - u(x))
plt.title("error between two NN and analytic solution")
plt.show()


# Below this we are try to get the extrapolation result which is not needed
# (interval a,b is given by integration limit is actually our domain)
# but still we are just plotting it for observation

"""
x = np.arange(-5.01, 5.01, 0.1)

z = np.outer(x, w_) + b_
s = sigmoid_(z)
N = s.dot(v_)
plt.plot(x, N, label='$NN_ solution$')
plt.plot(x, u(x), label='$Analytic solution$')
plt.xlabel('$x$')
plt.ylabel('solution')
plt.grid()
plt.legend()
plt.title("Extrapolation result for NN and its analytic solution")
plt.show()
"""


