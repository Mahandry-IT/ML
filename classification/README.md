# TP — Diagnostic de la Rouille Polysora sur Feuilles de Maïs

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Ordre d'exécution

| Fichier | Partie du TP | Rôle |
|---|---|---|
| `00_generate_synthetic_dataset.py` | (préparation) | Génère un dataset factice `dataset/saines/` et `dataset/malades/` pour pouvoir tester tout le pipeline. **À remplacer par de vraies photos.** |
| `01_feature_extraction.py` | Partie 1 | Extrait `pct_rouille`, `rugosite`, `score_dispersion_taches` de chaque image → `features_dataset.csv` |
| `02_max_minority.py` | Partie 2 | Métrique de pureté `P(t)` et fonction `trouver_meilleur_split` |
| `03_arbre_from_scratch.py` | Partie 3.1 | Arbre de décision Max-Minority récursif (`build_tree`) |
| `04_random_forest_from_scratch.py` | Partie 3.2 | Random Forest maison (bagging + vote majoritaire) |
| `05_comparaison_modeles.py` | Partie 3.3 | Entraîne et compare les 4 modèles, tableau Accuracy/Précision/Rappel |
| `06_sauvegarder_modele.py` | (avant Partie 4) | Sauvegarde le modèle final en `.pkl` pour l'app |
| `app.py` | Partie 4 | Application Streamlit (upload + prédiction + galerie) |

Pipeline complet :

```bash
python 00_generate_synthetic_dataset.py   # ou dépose tes vraies photos ici
python 01_feature_extraction.py
python 05_comparaison_modeles.py
python 06_sauvegarder_modele.py
streamlit run app.py
```

## 3. Ce que contient chaque partie (explication technique)

### Partie 1 — Feature Engineering
- **`pct_rouille` (X1)** : conversion BGR→HSV, puis masque sur la teinte
  "rouille" (H entre 5 et 30, orange/brun) restreint aux pixels de la
  feuille (filtrage par saturation/valeur pour ignorer un éventuel fond).
- **`rugosite` (X2)** : filtre de Sobel (dérivées en x et y) sur l'image en
  niveaux de gris, magnitude du gradient, puis variance de cette
  magnitude. Plus il y a de pustules, plus la texture est irrégulière,
  donc plus la variance est élevée.
- **`score_dispersion_taches` (X3, feature personnelle)** : compte le
  nombre de composantes connexes du masque "rouille" rapporté à leur
  surface totale. Intuition : la rouille se manifeste par de nombreuses
  petites pustules dispersées, alors qu'une simple ombre ou salissure
  forme une tache compacte unique — même à `pct_rouille` égal, la
  dispersion spatiale permet de trancher.

### Partie 2 — Indice Max-Minority
`P(t) = max(n0/N, n1/N)`. `trouver_meilleur_split` trie la variable,
teste tous les seuils candidats (milieux entre valeurs uniques
consécutives) et garde celui qui maximise la pureté pondérée du split.

### Partie 3 — Arbres et forêts from scratch vs scikit-learn
- `build_tree` récurse tant que le nœud n'est pas pur et que
  `max_depth` n'est pas atteinte, en choisissant à chaque nœud la
  meilleure variable/seuil parmi toutes les variables.
- `RandomForestMaxMinority` applique le bagging (tirage avec remplacement
  des lignes via `rng.choice`) et, à chaque nœud, ne teste qu'un
  sous-ensemble aléatoire des variables (`max_features="sqrt"`), comme
  scikit-learn, pour décorréler les arbres. La prédiction finale est un
  vote majoritaire.
- `05_comparaison_modeles.py` entraîne les 4 configurations sur le même
  split train/test et affiche Accuracy / Précision / Rappel + matrices de
  confusion.

**Sur le dataset synthétique fourni, les 4 modèles atteignent 100%** car
les classes y sont artificiellement très séparables. **Sur de vraies
photos, attends-toi à des scores plus modestes** (bruit d'éclairage,
angles de prise de vue, maladies concurrentes, etc.) — c'est normal et
c'est précisément ce qui rendra la comparaison entre les 4 modèles
intéressante à discuter.

### Discussion agronomique (à adapter à tes résultats réels)
- Un **Faux Négatif** (feuille malade jugée saine) laisse la maladie se
  propager → conséquence potentiellement lourde (perte de récolte).
- Un **Faux Positif** (feuille saine jugée malade) coûte un traitement
  inutile mais n'a pas de conséquence sanitaire.
- Le coût d'un Faux Négatif étant plus grave, **on privilégie un modèle à
  Rappel élevé** (quitte à sacrifier un peu de précision) : entre deux
  modèles à accuracy proche, choisis celui qui **minimise les FN** dans sa
  matrice de confusion. En général, la Random Forest (maison ou
  scikit-learn) est plus stable qu'un arbre unique car elle moyenne
  l'erreur de nombreux arbres décorrélés (bagging + sous-espace de
  variables), ce qui réduit la variance et le risque de sur-apprentissage
  sur un seuil de split malchanceux.

### Partie 4 — Application Streamlit
`app.py` réutilise directement les fonctions de `01_feature_extraction.py`
pour garantir que les features calculées en production sont identiques à
celles de l'entraînement. Le modèle est chargé une seule fois
(`@st.cache_resource`). Chaque image téléversée est copiée dans
`uploads/` avec un horodatage, et son diagnostic est stocké dans un
fichier `.label` à côté, ce qui permet à la galerie (partie 2 de l'app)
de reconstruire l'historique complet sans base de données.

## 4. Limites connues / pistes d'amélioration
- Le masque HSV de rouille est volontairement simple ; sur de vraies
  photos, affine les bornes de teinte selon tes propres échantillons.
- `score_dispersion_taches` peut être sensible au bruit sur des images
  de mauvaise qualité — un léger flou/érosion avant `connectedComponents`
  peut aider.
- Le dataset synthétique sert uniquement à valider le pipeline ; les
  résultats numériques n'ont de valeur qu'avec de vraies photos.
