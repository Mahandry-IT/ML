import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(2, 20, 200)
y = np.exp(x**2/2)

plt.plot(x, y, color='green', label='y = exp(x^2/2)')
plt.yscale('log')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title(r"$f(x) = e^{x^2/2}$")
plt.show()