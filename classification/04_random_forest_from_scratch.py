"""
04_random_forest_from_scratch.py
----------------------------------
PARTIE 3.2 du TP : Le Random Forest Max-Minority (fait maison).

Principe :
  1. BAGGING : pour chaque arbre, on tire N lignes AVEC REMPLACEMENT parmi
     les données d'entraînement (np.random.choice) -> chaque arbre voit un
     sous-échantillon légèrement différent -> diversité entre les arbres.
  2. (Bonus decorrelation) A chaque split, on peut aussi ne considérer
     qu'un sous-ensemble aléatoire des variables (comme scikit-learn) pour
     décorréler davantage les arbres. Ici, avec seulement 3 variables,
     on l'active en tirant 2 variables sur 3 à chaque noeud.
  3. VOTE MAJORITAIRE : la prédiction finale = classe majoritaire parmi les
     prédictions de tous les arbres de la forêt.
"""

import numpy as np
from importlib import import_module

arbre_mod = import_module("03_arbre_from_scratch")
Noeud = arbre_mod.Noeud
purete = import_module("02_max_minority").purete
trouver_meilleur_split = import_module("02_max_minority").trouver_meilleur_split


def _classe_majoritaire(y):
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    return 0 if n0 >= n1 else 1


def build_tree_avec_sous_espace(X, y, features_disponibles, depth=0,
                                 max_depth=5, min_samples_split=4,
                                 max_features=None, rng=None):
    """
    Variante de build_tree() qui, à chaque noeud, ne teste qu'un sous-
    ensemble aléatoire de `max_features` variables parmi `features_disponibles`
    (mécanisme classique de Random Forest pour décorréler les arbres).
    """
    n = len(y)
    p_noeud = purete(y)

    if p_noeud == 1.0 or depth >= max_depth or n < min_samples_split:
        return Noeud(prediction=_classe_majoritaire(y), purete_noeud=p_noeud, n_exemples=n)

    if max_features is None:
        candidats = features_disponibles
    else:
        k = min(max_features, len(features_disponibles))
        candidats = rng.choice(features_disponibles, size=k, replace=False)

    meilleure_variable, meilleur_seuil, meilleure_p_split = None, None, -1.0
    for j in candidats:
        seuil, p_split = trouver_meilleur_split(X[:, j], y)
        if seuil is None:
            continue
        if p_split > meilleure_p_split:
            meilleure_p_split, meilleure_variable, meilleur_seuil = p_split, j, seuil

    if meilleure_variable is None:
        return Noeud(prediction=_classe_majoritaire(y), purete_noeud=p_noeud, n_exemples=n)

    masque_gauche = X[:, meilleure_variable] <= meilleur_seuil
    X_g, y_g = X[masque_gauche], y[masque_gauche]
    X_d, y_d = X[~masque_gauche], y[~masque_gauche]

    if len(y_g) == 0 or len(y_d) == 0:
        return Noeud(prediction=_classe_majoritaire(y), purete_noeud=p_noeud, n_exemples=n)

    enfant_g = build_tree_avec_sous_espace(X_g, y_g, features_disponibles, depth + 1,
                                            max_depth, min_samples_split, max_features, rng)
    enfant_d = build_tree_avec_sous_espace(X_d, y_d, features_disponibles, depth + 1,
                                            max_depth, min_samples_split, max_features, rng)

    return Noeud(variable=meilleure_variable, seuil=meilleur_seuil,
                  gauche=enfant_g, droite=enfant_d,
                  purete_noeud=p_noeud, n_exemples=n)


class RandomForestMaxMinority:
    """Forêt aléatoire "from scratch" basée sur nos arbres Max-Minority."""

    def __init__(self, n_estimators=50, max_depth=5, min_samples_split=4,
                 max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.arbres_ = []
        self.n_features_ = None

    def _resoudre_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features is None:
            return n_features
        return int(self.max_features)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        max_feat = self._resoudre_max_features(n_features)
        features_disponibles = np.arange(n_features)

        rng = np.random.default_rng(self.random_state)
        self.arbres_ = []

        for i in range(self.n_estimators):
            # --- Bagging : tirage avec remplacement des lignes ---
            indices_bootstrap = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices_bootstrap], y[indices_bootstrap]

            arbre = build_tree_avec_sous_espace(
                X_boot, y_boot, features_disponibles,
                depth=0, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat, rng=rng,
            )
            self.arbres_.append(arbre)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # matrice (n_arbres, n_exemples) de prédictions individuelles
        toutes_predictions = np.array([
            arbre_mod.predire(arbre, X) for arbre in self.arbres_
        ])
        # vote majoritaire par colonne (par exemple)
        predictions_finales = (toutes_predictions.mean(axis=0) >= 0.5).astype(int)
        return predictions_finales

    def feature_importance(self):
        """Importance = fréquence d'utilisation de chaque variable dans les
        splits de tous les arbres de la forêt (mesure simple mais parlante)."""
        compteur = np.zeros(self.n_features_)

        def parcourir(noeud):
            if noeud is None or noeud.est_feuille():
                return
            compteur[noeud.variable] += 1
            parcourir(noeud.gauche)
            parcourir(noeud.droite)

        for arbre in self.arbres_:
            parcourir(arbre)

        total = compteur.sum()
        return compteur / total if total > 0 else compteur


if __name__ == "__main__":
    # Test rapide
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[0.05, 5], scale=[0.02, 1.5], size=(30, 2))
    X1 = rng.normal(loc=[0.45, 30], scale=[0.05, 3.0], size=(30, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 30 + [1] * 30)

    rf = RandomForestMaxMinority(n_estimators=25, max_depth=4)
    rf.fit(X, y)
    preds = rf.predict(X)
    acc = (preds == y).mean()
    print(f"Accuracy sur l'échantillon d'entraînement : {acc:.3f}")
    print("Importance des variables :", rf.feature_importance())
