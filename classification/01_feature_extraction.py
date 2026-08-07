"""
01_feature_extraction.py
-------------------------
PARTIE 1 du TP : Du Pixel aux Caractéristiques.

Pour chaque image du dataset (dataset/saines/, dataset/malades/), on extrait
un vecteur de 3 caractéristiques numériques :

  X1 = pct_rouille  -> % de pixels "couleur rouille" (masque HSV)
  X2 = rugosite     -> énergie moyenne du gradient de Sobel (texture)
  X3 = symetrie_contour -> feature PERSONNELLE (voir justification ci-dessous)

Le livrable est un DataFrame :
    [ID_Image | pct_rouille | rugosite | symetrie_contour | label_malade]
"""

import os
import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# X1 : pct_rouille (espace HSV)
# ----------------------------------------------------------------------
def extraire_pct_rouille(img_bgr):
    """
    Convertit l'image en HSV et calcule la proportion de pixels dont la
    teinte correspond à la rouille (orange/brun/jaune), PARMI les pixels
    appartenant réellement à la feuille (on exclut le fond via un masque
    de "feuille" basé sur la saturation/valeur, pour ne pas fausser le
    ratio avec un éventuel fond non-végétal).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Masque de la feuille = pixels suffisamment saturés/éclairés
    # (élimine un éventuel fond noir/blanc de la photo)
    masque_feuille = (s > 30) & (v > 30)
    total_feuille = max(int(masque_feuille.sum()), 1)

    # Plage de teinte "rouille" en OpenCV (H in [0,179])
    # oranges/bruns/jaunes ~ H entre 5 et 30
    masque_rouille = (h >= 5) & (h <= 30) & masque_feuille

    pct_rouille = masque_rouille.sum() / total_feuille
    return float(pct_rouille), masque_rouille


# ----------------------------------------------------------------------
# X2 : rugosite (filtre de Sobel)
# ----------------------------------------------------------------------
def extraire_rugosite(img_bgr):
    """
    Applique un filtre de Sobel (dérivées en x et y) sur l'image en niveaux
    de gris, puis calcule la magnitude du gradient en chaque pixel.
    La RUGOSITE = variance de cette magnitude : une feuille malade présente
    des pustules qui créent des variations d'intensité brusques et
    irrégulières -> variance plus élevée qu'une feuille saine, lisse.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # réduit le bruit avant Sobel

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    rugosite = float(np.var(magnitude))
    return rugosite


# ----------------------------------------------------------------------
# X3 : feature personnelle -> "score_dispersion_taches"
# ----------------------------------------------------------------------
def extraire_dispersion_taches(masque_rouille):
    """
    FEATURE PERSONNELLE : score_dispersion_taches

    Intuition agronomique : au stade précoce d'infection, la rouille
    apparaît sous forme de PETITES PUSTULES ÉPARSES sur toute la feuille,
    alors qu'une simple tache de salissure/ombre (faux-positif du masque
    couleur) forme en général UNE SEULE zone compacte. On distingue donc
    les deux cas non pas par la quantité de "orange" détecté, mais par
    son degré de FRAGMENTATION spatiale.

    Implémentation : on détecte les composantes connexes du masque_rouille
    (cv2.connectedComponents) et on calcule :
        score = nombre_de_composantes / (aire_totale_rouille + 1)
    -> Beaucoup de petites taches dispersées => score élevé (typique rouille)
    -> Une seule grosse tache compacte      => score faible (probable artefact)

    C'est une caractéristique complémentaire à pct_rouille : deux feuilles
    peuvent avoir le même pct_rouille mais une texture d'infection très
    différente, ce qui aide l'arbre de décision à mieux séparer les cas
    ambigus.
    """
    masque_uint8 = (masque_rouille.astype(np.uint8)) * 255
    n_composantes, labels = cv2.connectedComponents(masque_uint8)
    n_taches = max(n_composantes - 1, 0)  # -1 pour exclure le fond (label 0)
    aire_totale = int(masque_rouille.sum())

    score = n_taches / (aire_totale + 1) * 1000  # x1000 pour lisibilité
    return float(score)


# ----------------------------------------------------------------------
# Pipeline complet : parcours des dossiers + construction du DataFrame
# ----------------------------------------------------------------------
def extraire_features_image(chemin_image):
    img = cv2.imread(chemin_image)
    if img is None:
        return None

    pct_rouille, masque_rouille = extraire_pct_rouille(img)
    rugosite = extraire_rugosite(img)
    dispersion = extraire_dispersion_taches(masque_rouille)

    return pct_rouille, rugosite, dispersion


def construire_dataframe(dossier_saines="dataset/saines",
                          dossier_malades="dataset/malades"):
    lignes = []

    for dossier, label in [(dossier_saines, 0), (dossier_malades, 1)]:
        if not os.path.isdir(dossier):
            print(f"[ATTENTION] Dossier introuvable : {dossier}")
            continue
        for nom_fichier in sorted(os.listdir(dossier)):
            if not nom_fichier.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            chemin = os.path.join(dossier, nom_fichier)
            resultat = extraire_features_image(chemin)
            if resultat is None:
                print(f"[ATTENTION] Image illisible, ignorée : {chemin}")
                continue
            pct_rouille, rugosite, dispersion = resultat
            lignes.append({
                "ID_Image": nom_fichier,
                "pct_rouille": pct_rouille,
                "rugosite": rugosite,
                "score_dispersion_taches": dispersion,
                "label_malade": label,
            })

    df = pd.DataFrame(lignes)
    return df


if __name__ == "__main__":
    df = construire_dataframe()
    print(df.head(10))
    print(f"\nTotal images traitées : {len(df)}")
    df.to_csv("../data/features_dataset.csv", index=False)
    print("Fichier sauvegardé : features_dataset.csv")
