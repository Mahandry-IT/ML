"""
TP : Descente de gradient générique
====================================

Implémentation qui fonctionne pour :
- n'importe quelle fonction de coût (analytique OU calculée numériquement) ;
- n'importe quel nombre de variables (features) ;
- le gradient utilisé est celui vu en cours, avec le facteur 1/(2m) dans le coût
  (donc 1/m dans le gradient, comme démontré dans le cours).

Autorisé : Python pur / numpy / matplotlib.
Interdit : scikit-learn, PyTorch, TensorFlow. (aucune de ces librairies n'est utilisée ici)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Fonction de coût générique : MSE avec le facteur 1/(2m) du cours
# ---------------------------------------------------------------------------
def mse_cost(theta, X, y):
    """
    Coût quadratique moyen (celui du cours) :
        J(theta) = 1/(2m) * somme( (X @ theta - y)^2 )

    theta : vecteur de paramètres, shape (n,)
    X     : matrice des features (avec colonne de biais incluse), shape (m, n)
    y     : vecteur des valeurs réelles, shape (m,)
    """
    m = len(y)
    erreurs = X @ theta - y
    return (1 / (2 * m)) * np.sum(erreurs ** 2)


# ---------------------------------------------------------------------------
# 2. Gradient analytique correspondant (formule matricielle du cours)
# ---------------------------------------------------------------------------
def mse_gradient(theta, X, y):
    """
    Gradient analytique de mse_cost :
        grad = 1/m * X^T @ (X @ theta - y)

    Cohérent avec le 1/(2m) du coût : la dérivée du carré fait apparaître un 2
    qui se simplifie avec le 1/(2m) -> il reste bien 1/m.
    """
    m = len(y)
    erreurs = X @ theta - y
    return (1 / m) * (X.T @ erreurs)


# ---------------------------------------------------------------------------
# 3. Gradient numérique (différences finies) : pour "n'importe quelle" fonction
#    de coût, même si on n'a pas sa dérivée analytique.
# ---------------------------------------------------------------------------
def numerical_gradient(cost_function, theta, X, y, epsilon=1e-5):
    """
    Approxime le gradient de cost_function en theta par différences finies
    centrées :
        d/d theta_i J ~= (J(theta + eps*e_i) - J(theta - eps*e_i)) / (2*eps)

    Permet d'utiliser gradient_descent() avec une fonction de coût quelconque,
    sans avoir à calculer sa dérivée à la main.
    """
    grad = np.zeros_like(theta, dtype=float)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_moins = theta.copy()
        theta_plus[i] += epsilon
        theta_moins[i] -= epsilon
        grad[i] = (
            cost_function(theta_plus, X, y) - cost_function(theta_moins, X, y)
        ) / (2 * epsilon)
    return grad


# ---------------------------------------------------------------------------
# 4. Algorithme de descente de gradient générique
# ---------------------------------------------------------------------------
def gradient_descent(
    X,
    y,
    cost_function=mse_cost,
    gradient_function=None,
    alpha=0.01,
    n_iterations=1000,
    theta_init=None,
    verbose=False,
    print_every=100,
):
    """
    Descente de gradient générique.

    Paramètres
    ----------
    X : array (m, n)          -> matrice des features (ajouter la colonne de biais
                                  avec add_intercept() si besoin d'un terme b)
    y : array (m,)             -> valeurs cibles
    cost_function(theta, X, y) -> fonction de coût à minimiser (défaut : mse_cost)
    gradient_function(theta, X, y) -> gradient de cost_function.
                                  Si None, un gradient numérique est utilisé
                                  automatiquement (fonctionne pour N'IMPORTE
                                  QUELLE fonction de coût).
    alpha : float               -> taux d'apprentissage
    n_iterations : int          -> nombre d'itérations
    theta_init : array (n,)     -> valeurs initiales des paramètres (défaut : zéros)

    Retour
    ------
    theta   : paramètres appris
    history : liste des valeurs de coût à chaque itération (pour tracer la courbe)
    """
    m, n = X.shape

    theta = np.zeros(n) if theta_init is None else np.array(theta_init, dtype=float)

    # Si aucune dérivée analytique n'est fournie, on calcule le gradient
    # numériquement : l'algorithme marche alors avec N'IMPORTE QUELLE cost_function.
    if gradient_function is None:
        gradient_function = lambda th, X, y: numerical_gradient(cost_function, th, X, y)

    history = []
    for it in range(n_iterations):
        grad = gradient_function(theta, X, y)
        theta = theta - alpha * grad          # w = w - alpha * dw  (formule du cours)

        cost = cost_function(theta, X, y)
        history.append(cost)

        if verbose and (it % print_every == 0 or it == n_iterations - 1):
            print(f"itération {it:4d} | coût = {cost:.6f}")

    return theta, history


# ---------------------------------------------------------------------------
# 5. Utilitaire : ajouter la colonne de biais (x0 = 1) pour intégrer b dans theta
# ---------------------------------------------------------------------------
def add_intercept(X):
    """Ajoute une colonne de 1 devant X pour que theta[0] joue le rôle du biais b."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # vecteur (m,) -> matrice colonne (m, 1)
    m = X.shape[0]
    return np.c_[np.ones(m), X]


# ===========================================================================
# DÉMONSTRATIONS (exemples du cours)
# ===========================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # Cas 1 : une seule variable (exemple du cours : x -> y = 2x + 1)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("Cas 1 : régression linéaire à UNE variable")
    print("=" * 60)

    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 7, 9])

    X = add_intercept(x)  # X devient [[1, 1], [1, 2], [1, 3], [1, 4]]

    theta, history = gradient_descent(
        X, y,
        cost_function=mse_cost,
        gradient_function=mse_gradient,   # gradient analytique du cours
        alpha=0.1,
        n_iterations=1000,
        verbose=True,
        print_every=200,
    )
    b, w = theta
    print(f"\nRésultat -> w = {w:.4f}, b = {b:.4f}  (attendu : w=2, b=1)")

    # -----------------------------------------------------------------
    # Cas 2 : plusieurs variables (surface, chambres, distance -> prix)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Cas 2 : régression linéaire à PLUSIEURS variables")
    print("=" * 60)

    X_multi = np.array([
        [50, 2, 3],
        [80, 3, 7],
        [120, 4, 10],
    ], dtype=float)
    y_multi = np.array([150000, 210000, 300000], dtype=float)

    # Normalisation (recommandée dans le cours : évite les zigzags)
    X_mean = X_multi.mean(axis=0)
    X_std = X_multi.std(axis=0)
    X_multi_norm = (X_multi - X_mean) / X_std

    X_multi_final = add_intercept(X_multi_norm)

    theta_multi, history_multi = gradient_descent(
        X_multi_final, y_multi,
        cost_function=mse_cost,
        gradient_function=mse_gradient,
        alpha=0.1,
        n_iterations=1000,
        verbose=True,
        print_every=200,
    )
    print(f"\nParamètres appris (sur données normalisées) : {theta_multi}")
    print(f"Coût final : {history_multi[-1]:.2f}")

    # -----------------------------------------------------------------
    # Cas 3 : vérifier que ça marche aussi SANS gradient analytique
    # (gradient numérique automatique -> "n'importe quelle" fonction de coût)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Cas 3 : même problème (cas 1), mais avec gradient NUMÉRIQUE")
    print("=" * 60)

    theta_num, history_num = gradient_descent(
        X, y,
        cost_function=mse_cost,
        gradient_function=None,  # -> déclenche le calcul numérique automatique
        alpha=0.1,
        n_iterations=1000,
        verbose=True,
        print_every=200,
    )
    b_num, w_num = theta_num
    print(f"\nRésultat (gradient numérique) -> w = {w_num:.4f}, b = {b_num:.4f}")

    # -----------------------------------------------------------------
    # (Optionnel) Tracer la courbe de convergence du coût
    # -----------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(history, label="Cas 1 (1 variable)")
        plt.plot(history_multi, label="Cas 2 (plusieurs variables)")
        plt.xlabel("Itération")
        plt.ylabel("Coût J(theta)")
        plt.title("Convergence de la descente de gradient")
        plt.legend()
        plt.tight_layout()
        plt.savefig("courbe_convergence.png", dpi=120)
        print("\nCourbe de convergence sauvegardée : courbe_convergence.png")
    except ImportError:
        pass
