import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,106,85,86])
colors = np.where(y > 100, 'red', 'blue')

plt.scatter(x, y, c=colors)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter plot with conditional coloring")
plt.show()