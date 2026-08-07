"""
05_comparaison_modeles.py
---------------------------
PARTIE 3.3 du TP : Comparaison et Analyse Critique.

Entraîne et évalue 4 configurations sur le MEME split train(80%)/test(20%) :
  1. Arbre "From Scratch" (Max-Minority)
  2. Random Forest "From Scratch" (Max-Minority)
  3. DecisionTreeClassifier (scikit-learn, critère Gini)
  4. RandomForestClassifier (scikit-learn, n_estimators=100)

Puis affiche un tableau comparatif Accuracy / Précision / Rappel et les
matrices de confusion, pour répondre aux questions d'analyse du TP.
"""

import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              confusion_matrix)

arbre_mod = import_module("03_arbre_from_scratch")
rf_mod = import_module("04_random_forest_from_scratch")

FEATURES = ["pct_rouille", "rugosite", "score_dispersion_taches"]


def charger_donnees(chemin_csv="../data/features_dataset.csv"):
    df = pd.read_csv(chemin_csv)
    X = df[FEATURES].values
    y = df["label_malade"].values
    return X, y, df


def evaluer(nom, y_test, y_pred, resultats):
    resultats.append({
        "Modele": nom,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Rappel": recall_score(y_test, y_pred, zero_division=0),
    })
    print(f"\n--- {nom} ---")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Rappel    : {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print("Matrice de confusion [[TN FP][FN TP]] :")
    print(confusion_matrix(y_test, y_pred))


def main():
    X, y, df = charger_donnees()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train : {len(X_train)} exemples | Test : {len(X_test)} exemples")

    resultats = []

    # 1) Arbre From Scratch (Max-Minority) -----------------------------
    arbre_maison = arbre_mod.build_tree(X_train, y_train, max_depth=5)
    pred_arbre_maison = arbre_mod.predire(arbre_maison, X_test)
    evaluer("Arbre From Scratch (Max-Minority)", y_test, pred_arbre_maison, resultats)

    # 2) Random Forest From Scratch (Max-Minority) ----------------------
    rf_maison = rf_mod.RandomForestMaxMinority(
        n_estimators=50, max_depth=5, max_features="sqrt", random_state=42
    )
    rf_maison.fit(X_train, y_train)
    pred_rf_maison = rf_maison.predict(X_test)
    evaluer("Random Forest From Scratch (Max-Minority)", y_test, pred_rf_maison, resultats)

    # 3) DecisionTreeClassifier (scikit-learn, Gini) ---------------------
    dt_sklearn = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    pred_dt_sklearn = dt_sklearn.predict(X_test)
    evaluer("DecisionTree scikit-learn (Gini)", y_test, pred_dt_sklearn, resultats)

    # 4) RandomForestClassifier (scikit-learn) ---------------------------
    rf_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sklearn.fit(X_train, y_train)
    pred_rf_sklearn = rf_sklearn.predict(X_test)
    evaluer("RandomForest scikit-learn (n=100)", y_test, pred_rf_sklearn, resultats)

    # --- Tableau comparatif final ---
    df_resultats = pd.DataFrame(resultats)
    print("\n\n================ TABLEAU COMPARATIF ================")
    print(df_resultats.to_string(index=False))
    df_resultats.to_csv("../data/resultats_comparaison.csv", index=False)

    # --- Importance des variables (analyse agronomique) ---
    print("\nImportance des variables (RF From Scratch) :")
    for nom_var, imp in zip(FEATURES, rf_maison.feature_importance()):
        print(f"  {nom_var:25s} : {imp:.3f}")

    print("\nImportance des variables (RF scikit-learn) :")
    for nom_var, imp in zip(FEATURES, rf_sklearn.feature_importances_):
        print(f"  {nom_var:25s} : {imp:.3f}")

    return df_resultats


if __name__ == "__main__":
    main()
