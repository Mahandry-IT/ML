"""
06_sauvegarder_modele.py
--------------------------
Entraîne le modèle retenu pour le déploiement (voir analyse de la partie
3.3 : on recommande en général le RandomForestClassifier scikit-learn pour
sa robustesse et sa disponibilité de predict_proba) sur 100% des données
disponibles, puis le sauvegarde en pickle pour que app.py puisse le charger.
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["pct_rouille", "rugosite", "score_dispersion_taches"]

df = pd.read_csv("../data/features_dataset.csv")
X = df[FEATURES].values
y = df["label_malade"].values

modele_final = RandomForestClassifier(n_estimators=100, random_state=42)
modele_final.fit(X, y)

with open("models/modele_final.pkl", "wb") as f:
    pickle.dump(modele_final, f)

print("Modèle entraîné sur", len(X), "images et sauvegardé -> models/modele_final.pkl")
