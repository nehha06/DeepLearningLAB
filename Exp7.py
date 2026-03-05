import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-8,8,0.1)

plt.plot(x,sigmoid(x))
plt.title("Sigmoid Curve")
plt.show()
