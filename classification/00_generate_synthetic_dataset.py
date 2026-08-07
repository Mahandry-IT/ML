"""
00_generate_synthetic_dataset.py
---------------------------------
Le TP demande de partir de VRAIES photos de terrain (dataset/saines/ et
dataset/malades/). Comme je n'ai pas accès à un jeu de photos réel ici,
ce script génère un jeu de données SYNTHÉTIQUE mais réaliste (feuilles
vertes + "pustules" orangées générées proceduralement) afin que TOUT le
pipeline (Partie 1 -> Partie 4) soit exécutable et testable de bout en bout.

>>> À FAIRE PAR TOI (obligatoire pour la remise du TP) <<<
Remplace le contenu de dataset/saines/ et dataset/malades/ par de vraies
photos (prises au téléphone, ou trouvées sur des jeux de données publics
type PlantVillage / Kaggle "Corn Leaf Disease"). Le reste du code
(Parties 1 à 4) n'a besoin d'AUCUNE modification : il lit simplement les
images depuis ces deux dossiers.
"""

import os
import numpy as np
import cv2

RNG = np.random.default_rng(42)

OUT_SAINES = "dataset/saines"
OUT_MALADES = "dataset/malades"
IMG_SIZE = 300
N_PER_CLASSE = 60  # 60 saines + 60 malades = 120 images


def fond_feuille(size=IMG_SIZE):
    """Crée une texture de fond vert (feuille), avec nervures et léger bruit,
    pour simuler une vraie photo de feuille de maïs prise au champ."""
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Dégradé de vert (variation naturelle de chlorophylle)
    base_h = RNG.integers(35, 45)      # teinte verte (HSV) réaliste
    for y in range(size):
        s = 120 + int(40 * np.sin(y / size * np.pi))
        v = 90 + int(60 * (y / size))
        img[y, :] = (base_h, s, v)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # Nervures (lignes plus claires)
    for _ in range(RNG.integers(3, 6)):
        x0 = RNG.integers(0, size)
        cv2.line(img, (x0, 0), (x0 + RNG.integers(-20, 20), size),
                  (200, 230, 200), 1, lineType=cv2.LINE_AA)

    # Bruit de capteur / texture naturelle
    bruit = RNG.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + bruit, 0, 255).astype(np.uint8)
    return img


def ajouter_pustules(img, intensite):
    """Ajoute des pustules orangées/brunâtres de rouille.
    intensite in [0,1] contrôle le pourcentage de surface touchée
    (0 = feuille saine, valeur haute = feuille très malade)."""
    size = img.shape[0]
    n_pustules = int(intensite * 90)
    for _ in range(n_pustules):
        cx, cy = RNG.integers(0, size), RNG.integers(0, size)
        r = RNG.integers(2, 6)
        # Teinte rouille : orange/brun -> BGR approx
        couleur = (
            int(RNG.integers(10, 40)),   # B
            int(RNG.integers(70, 130)),  # G
            int(RNG.integers(150, 220)), # R
        )
        cv2.circle(img, (cx, cy), r, couleur, -1, lineType=cv2.LINE_AA)
    # Léger flou pour rendre les pustules moins "parfaites"
    if n_pustules > 0:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generer_dataset():
    os.makedirs(OUT_SAINES, exist_ok=True)
    os.makedirs(OUT_MALADES, exist_ok=True)

    for i in range(N_PER_CLASSE):
        # --- Feuille saine : pas ou très peu de pustules (bruit naturel) ---
        img = fond_feuille()
        img = ajouter_pustules(img, intensite=RNG.uniform(0.0, 0.05))
        cv2.imwrite(f"{OUT_SAINES}/saine_{i:03d}.jpg", img)

        # --- Feuille malade : pustules nombreuses, sévérité variable ---
        img2 = fond_feuille()
        img2 = ajouter_pustules(img2, intensite=RNG.uniform(0.25, 0.9))
        cv2.imwrite(f"{OUT_MALADES}/malade_{i:03d}.jpg", img2)

    print(f"[OK] {N_PER_CLASSE} images saines -> {OUT_SAINES}/")
    print(f"[OK] {N_PER_CLASSE} images malades -> {OUT_MALADES}/")


if __name__ == "__main__":
    generer_dataset()
