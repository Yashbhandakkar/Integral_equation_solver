import numpy as np
"""
x = np.arange(0, 1.01, 0.01)
print(x.shape)
print(np.pi/2)
x = np.arange(0, np.pi/2, 0.1)
print(x)
"""
a = 0
b = np.pi/2
x = np.arange(a, b, 0.1)
length = len(x)
print(x)


for i in range(0, length):
    kk = x[i]
    t = float(kk)
    print(t)


