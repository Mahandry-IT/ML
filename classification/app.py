"""
app.py
------
PARTIE 4 du TP : Déploiement d'une Application Web avec Streamlit.

Lancer avec :   streamlit run app.py

Fonctionnalités :
  1. Upload + prédiction en temps réel (st.file_uploader)
  2. Galerie d'historique des détections (st.columns)

L'app réutilise TELLES QUELLES les fonctions de la Partie 1
(01_feature_extraction.py) pour garantir que le modèle voit exactement
les mêmes features à l'entraînement et en production.
"""

import os
import pickle
import glob
from datetime import datetime
from importlib import import_module

import cv2
import numpy as np
import streamlit as st

fe = import_module("01_feature_extraction")

MODEL_PATH = "models/modele_final.pkl"
UPLOAD_DIR = "uploads"
FEATURES_ORDER = ["pct_rouille", "rugosite", "score_dispersion_taches"]

st.set_page_config(page_title="Diagnostic Rouille du Maïs", page_icon="🌽", layout="wide")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Chargement du modèle entraîné (pickle)
# ----------------------------------------------------------------------
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predire_image(modele, chemin_image):
    """Extrait les features (Partie 1) puis prédit avec le modèle chargé."""
    resultat = fe.extraire_features_image(chemin_image)
    if resultat is None:
        return None, None
    pct_rouille, rugosite, dispersion = resultat
    X = np.array([[pct_rouille, rugosite, dispersion]])

    # Compat : modèle scikit-learn (a .predict) OU arbre/forêt maison
    # (module 03/04, qui a une fonction predire() séparée) -> on détecte
    # dynamiquement le bon appel.
    if hasattr(modele, "predict"):
        pred = int(modele.predict(X)[0])
    else:
        arbre_mod = import_module("03_arbre_from_scratch")
        pred = int(arbre_mod.predire(modele, X)[0])

    features = {"pct_rouille": pct_rouille, "rugosite": rugosite,
                "score_dispersion_taches": dispersion}
    return pred, features


# ----------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------
st.title("🌽 Diagnostic de la Rouille Polysora sur Feuilles de Maïs")
st.caption("Outil d'aide au diagnostic pour techniciens agricoles — Madagascar")

modele = charger_modele()
if modele is None:
    st.error(
        f"Aucun modèle trouvé à `{MODEL_PATH}`. "
        "Entraîne d'abord un modèle (voir 05_comparaison_modeles.py) "
        "et sauvegarde-le avec pickle dans ce chemin."
    )

st.header("1. Analyser une nouvelle feuille")
fichier = st.file_uploader("Téléverse une photo de feuille de maïs",
                            type=["png", "jpg", "jpeg"])

if fichier is not None and modele is not None:
    # Sauvegarde locale (pour pouvoir la relire avec cv2 et l'archiver)
    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_sauvegarde = f"{horodatage}_{fichier.name}"
    chemin_sauvegarde = os.path.join(UPLOAD_DIR, nom_sauvegarde)
    with open(chemin_sauvegarde, "wb") as f:
        f.write(fichier.getbuffer())

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(chemin_sauvegarde, caption="Feuille téléversée", use_container_width=True)

    with col_result:
        with st.spinner("Extraction des caractéristiques et prédiction..."):
            pred, features = predire_image(modele, chemin_sauvegarde)

        if pred is None:
            st.error("Image illisible, réessaie avec un autre fichier.")
        else:
            st.subheader("Résultat du diagnostic")
            if pred == 1:
                st.error("⚠️ ATTENTION : Feuille Malade (Rouille détectée)")
            else:
                st.success("✅ Feuille Saine")

            st.write("**Caractéristiques extraites :**")
            st.json(features)

            # On enregistre le diagnostic à côté de l'image (fichier .txt
            # simple, pour que la galerie puisse relire le résultat)
            with open(chemin_sauvegarde + ".label", "w") as f:
                f.write(str(pred))

st.divider()

# ----------------------------------------------------------------------
# 2. Galerie d'historique des détections
# ----------------------------------------------------------------------
st.header("2. Historique des détections")

fichiers_images = sorted(
    glob.glob(os.path.join(UPLOAD_DIR, "*.jpg"))
    + glob.glob(os.path.join(UPLOAD_DIR, "*.jpeg"))
    + glob.glob(os.path.join(UPLOAD_DIR, "*.png")),
    reverse=True,  # les plus récentes en premier
)

if not fichiers_images:
    st.info("Aucune image analysée pour le moment.")
else:
    NB_COLONNES = 4
    colonnes = st.columns(NB_COLONNES)

    for i, chemin_img in enumerate(fichiers_images):
        chemin_label = chemin_img + ".label"
        if os.path.exists(chemin_label):
            with open(chemin_label) as f:
                label = f.read().strip()
            diagnostic = "🔴 Malade" if label == "1" else "🟢 Saine"
        else:
            diagnostic = "❔ Non diagnostiquée"

        with colonnes[i % NB_COLONNES]:
            st.image(chemin_img, use_container_width=True)
            st.caption(f"{os.path.basename(chemin_img)}\n{diagnostic}")
