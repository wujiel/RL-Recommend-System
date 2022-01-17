# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from actor import ActorNetwork
actor = ActorNetwork(100,128)


mean = (1, 1)
cov = np.array([[0.1, 0], [0, 1]])
x = np.random.multivariate_normal(mean, cov, 'raise')  # nx2

plt.scatter(x[:, 0], x[:, 1])
plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
