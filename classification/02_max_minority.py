"""
02_max_minority.py
-------------------
PARTIE 2 du TP : Le Défi Algorithmique - L'Indice "Max-Minority".

Pureté d'un noeud t (N individus, classes 0/1) :
    P(t) = max_c ( n_c / N )

Pureté pondérée d'un split (G = gauche, D = droite) :
    P_split = (|G|/N) * P(G) + (|D|/N) * P(D)

On cherche le seuil s qui MAXIMISE P_split (plus P_split est proche de 1,
plus le split sépare bien les deux classes).
"""

import numpy as np


def purete(y):
    """P(t) = proportion de la classe majoritaire dans y."""
    n = len(y)
    if n == 0:
        return 0.0
    y = np.asarray(y)
    n_classe_0 = np.sum(y == 0)
    n_classe_1 = np.sum(y == 1)
    return max(n_classe_0, n_classe_1) / n


def trouver_meilleur_split(X_column, y):
    """
    Teste tous les seuils candidats pour une variable continue X_column
    et retourne (meilleur_seuil, meilleure_purete_ponderee).

    X_column : array-like, valeurs de la variable à tester (ex: pct_rouille)
    y        : array-like, labels correspondants (0 ou 1)
    """
    X_column = np.asarray(X_column, dtype=float)
    y = np.asarray(y)
    N = len(y)

    # --- Etape 1 : trier les données par X_column ---
    ordre = np.argsort(X_column)
    X_trie = X_column[ordre]
    y_trie = y[ordre]

    # --- Etape 2 : initialiser le meilleur seuil et la meilleure pureté ---
    meilleur_seuil = None
    meilleure_purete = -1.0

    # Valeurs uniques triées -> seuils candidats = milieux entre 2 valeurs
    # uniques consécutives (inutile de tester un seuil entre deux valeurs
    # identiques, le split serait vide d'un côté).
    valeurs_uniques = np.unique(X_trie)
    if len(valeurs_uniques) < 2:
        # Impossible de splitter (toutes les valeurs sont identiques)
        return None, purete(y_trie)

    # --- Etape 3 : parcourir les seuils candidats ---
    for i in range(len(valeurs_uniques) - 1):
        seuil = (valeurs_uniques[i] + valeurs_uniques[i + 1]) / 2.0

        masque_gauche = X_trie <= seuil
        y_gauche = y_trie[masque_gauche]
        y_droite = y_trie[~masque_gauche]

        if len(y_gauche) == 0 or len(y_droite) == 0:
            continue  # split dégénéré, on l'ignore

        p_gauche = purete(y_gauche)
        p_droite = purete(y_droite)

        p_split = (len(y_gauche) / N) * p_gauche + (len(y_droite) / N) * p_droite

        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil = seuil

    # --- Etape 4 : retourner le seuil optimal et sa pureté associée ---
    return meilleur_seuil, meilleure_purete


if __name__ == "__main__":
    # Petit test unitaire "à la main" pour vérifier le comportement
    # Exemple : pct_rouille bien séparateur entre saines (~0) et malades (~0.4)
    X_test = np.array([0.01, 0.02, 0.03, 0.35, 0.40, 0.55])
    y_test = np.array([0, 0, 0, 1, 1, 1])

    seuil, p = trouver_meilleur_split(X_test, y_test)
    print(f"Meilleur seuil trouvé : {seuil:.4f}")
    print(f"Pureté pondérée du split : {p:.4f}  (1.0 = séparation parfaite)")
