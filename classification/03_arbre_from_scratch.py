"""
03_arbre_from_scratch.py
--------------------------
PARTIE 3.1 du TP : L'Arbre de Décision Max-Minority (fait maison).

On construit récursivement un arbre binaire :
 - à chaque noeud, on teste TOUTES les variables (pct_rouille, rugosite,
   score_dispersion_taches) avec trouver_meilleur_split() et on garde la
   variable + le seuil qui donnent la meilleure P_split.
 - on s'arrête si :
     * le noeud est 100% pur (P(t) == 1), ou
     * la profondeur maximale max_depth est atteinte, ou
     * il n'y a plus assez d'exemples / plus de split possible.
 - une feuille prédit la classe majoritaire de ses exemples.
"""

import numpy as np
from importlib import import_module

# 02_max_minority.py n'est pas un nom de module Python valide (commence par
# un chiffre) -> on l'importe dynamiquement.
mm = import_module("02_max_minority")
purete = mm.purete
trouver_meilleur_split = mm.trouver_meilleur_split


class Noeud:
    """Un noeud de l'arbre : soit une feuille (prediction != None),
    soit un noeud interne (variable/seuil/enfants gauche+droite)."""

    def __init__(self, prediction=None, variable=None, seuil=None,
                 gauche=None, droite=None, purete_noeud=None, n_exemples=None):
        self.prediction = prediction      # classe prédite si feuille
        self.variable = variable          # index de la colonne utilisée
        self.seuil = seuil                # seuil de split
        self.gauche = gauche              # sous-arbre gauche (Noeud)
        self.droite = droite              # sous-arbre droite (Noeud)
        self.purete_noeud = purete_noeud  # P(t) de ce noeud (pour analyse)
        self.n_exemples = n_exemples

    def est_feuille(self):
        return self.prediction is not None


def _classe_majoritaire(y):
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    return 0 if n0 >= n1 else 1


def build_tree(X, y, depth=0, max_depth=5, min_samples_split=4):
    """
    Construit récursivement l'arbre Max-Minority.

    X : np.ndarray shape (n_samples, n_features)
    y : np.ndarray shape (n_samples,)
    depth : profondeur courante (0 à la racine)
    max_depth : profondeur maximale autorisée
    min_samples_split : nb minimal d'exemples pour tenter un split
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    p_noeud = purete(y)

    # --- Conditions d'arrêt ---
    if p_noeud == 1.0 or depth >= max_depth or n < min_samples_split:
        return Noeud(prediction=_classe_majoritaire(y),
                      purete_noeud=p_noeud, n_exemples=n)

    # --- Chercher le meilleur split, toutes variables confondues ---
    meilleure_variable = None
    meilleur_seuil = None
    meilleure_purete_split = -1.0

    n_features = X.shape[1]
    for j in range(n_features):
        seuil, p_split = trouver_meilleur_split(X[:, j], y)
        if seuil is None:
            continue
        if p_split > meilleure_purete_split:
            meilleure_purete_split = p_split
            meilleure_variable = j
            meilleur_seuil = seuil

    # Aucun split valide trouvé (ex: toutes les variables constantes)
    if meilleure_variable is None:
        return Noeud(prediction=_classe_majoritaire(y),
                      purete_noeud=p_noeud, n_exemples=n)

    # --- Split effectif des données ---
    masque_gauche = X[:, meilleure_variable] <= meilleur_seuil
    X_g, y_g = X[masque_gauche], y[masque_gauche]
    X_d, y_d = X[~masque_gauche], y[~masque_gauche]

    # Sécurité : si un split ne sépare rien réellement, on arrête ici
    if len(y_g) == 0 or len(y_d) == 0:
        return Noeud(prediction=_classe_majoritaire(y),
                      purete_noeud=p_noeud, n_exemples=n)

    # --- Récursion sur les deux sous-groupes ---
    enfant_gauche = build_tree(X_g, y_g, depth + 1, max_depth, min_samples_split)
    enfant_droite = build_tree(X_d, y_d, depth + 1, max_depth, min_samples_split)

    return Noeud(variable=meilleure_variable, seuil=meilleur_seuil,
                  gauche=enfant_gauche, droite=enfant_droite,
                  purete_noeud=p_noeud, n_exemples=n)


def predire_un(noeud, x):
    """Parcourt l'arbre pour un seul exemple x (1D array) et retourne 0/1."""
    if noeud.est_feuille():
        return noeud.prediction
    if x[noeud.variable] <= noeud.seuil:
        return predire_un(noeud.gauche, x)
    else:
        return predire_un(noeud.droite, x)


def predire(noeud, X):
    """Prédit une classe pour chaque ligne de X (2D array)."""
    X = np.asarray(X, dtype=float)
    return np.array([predire_un(noeud, x) for x in X])


def afficher_arbre(noeud, noms_variables, profondeur=0):
    """Affiche l'arbre en texte, pour l'inspection pédagogique."""
    prefixe = "  " * profondeur
    if noeud.est_feuille():
        print(f"{prefixe}Feuille -> classe {noeud.prediction} "
              f"(n={noeud.n_exemples}, purete={noeud.purete_noeud:.2f})")
        return
    nom_var = noms_variables[noeud.variable]
    print(f"{prefixe}[{nom_var} <= {noeud.seuil:.4f}] "
          f"(n={noeud.n_exemples}, purete={noeud.purete_noeud:.2f})")
    afficher_arbre(noeud.gauche, noms_variables, profondeur + 1)
    afficher_arbre(noeud.droite, noms_variables, profondeur + 1)


if __name__ == "__main__":
    # Test rapide sur un jeu jouet
    X_test = np.array([
        [0.01, 5.0], [0.02, 6.0], [0.03, 4.5],
        [0.40, 30.0], [0.45, 28.0], [0.50, 33.0],
    ])
    y_test = np.array([0, 0, 0, 1, 1, 1])

    arbre = build_tree(X_test, y_test, max_depth=3)
    afficher_arbre(arbre, noms_variables=["pct_rouille", "rugosite"])

    preds = predire(arbre, X_test)
    print("Prédictions :", preds)
    print("Réel        :", y_test)
