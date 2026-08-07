import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# exo 1
n = np.linspace(1, 4, 100)

functions = {
    "f(n) = n!": [math.gamma(i + 1) for i in n],
    "f(n) = e^n": np.exp(n),
    "f(n) = n^2": n**2,
    "f(n) = 3^(log_2(n))": 3**np.log2(n),
    "f(n) = n^(3/2)": n**(3/2),
    "f(n) = n*log(n)": n*np.log(n),
    "f(n) = n": n,
    "f(n) = sqrt(n)": np.sqrt(n),
    "f(n) = ln(n)": np.log(n),
    "f(n) = 3": 3 + 0*n,
}

n_large = 100
asymptotic_values = {
    "f(n) = n!": math.gamma(n_large + 1),
    "f(n) = e^n": np.exp(n_large),
    "f(n) = n^2": n_large**2,
    "f(n) = 3^(log_2(n))": 3**np.log2(n_large),
    "f(n) = n^(3/2)": n_large**1.5,
    "f(n) = n*log(n)": n_large*np.log(n_large),
    "f(n) = n": n_large,
    "f(n) = sqrt(n)": np.sqrt(n_large),
    "f(n) = ln(n)": np.log(n_large),
    "f(n) = 3": 3,
}

ordered_functions = sorted(
    functions.items(),
    key=lambda item: asymptotic_values[item[0]],
    reverse=True
)

colors = ['red', 'green', 'blue', 'orange', 'purple',
          'brown', 'pink', 'gray', 'olive', 'cyan']

# --- Un seul graphe (meme plt), une couleur par fonction ---
plt.figure(figsize=(10, 6))
for (label, y), color in zip(ordered_functions, colors):
    plt.plot(n, y, color=color, label=label)

plt.xlabel("n")
plt.ylabel("f(n)")
plt.yscale("log")  # indispensable : les echelles varient de 1 a des milliers
plt.title("Comparaison des fonctions - ordre de croissance vers l'infini")
plt.legend()
plt.tight_layout()
plt.show()

print("Ordre de grandeur (du plus grand au plus petit, n -> infini) :")
for label, _ in ordered_functions:
    print(" -", label)