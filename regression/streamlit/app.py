import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Chargement du modèle et des objets de prétraitement ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
colonnes_modele = joblib.load("colonnes_modele.pkl")
colonnes_a_standardiser = joblib.load("colonnes_a_standardiser.pkl")

st.set_page_config(page_title="Prédiction de loyer - Antananarivo", page_icon="🏠")
st.title("🏠 Prédiction du loyer mensuel à Antananarivo")
st.write("Renseigne les caractéristiques du logement pour estimer son loyer mensuel.")

# --- Formulaire de saisie utilisateur ---
quartier = st.selectbox(
    "Quartier",
    [
        "67 Ha",
        "Alakamisy Fenoarivo",
        "Alarobia",
        "Ambanidia",
        "Ambatobe",
        "Ambatoroka",
        "Ambodivona",
        "Ambohidahy",
        "Ambohimanga",
        "Ambohimangakely",
        "Ambohijatovo",
        "Ambohipo",
        "Ampefiloha",
        "Ampandrana",
        "Ampasampito",
        "Ampitatafika",
        "Analakely",
        "Andavamamba",
        "Andohalo",
        "Andraharo",
        "Andravoahangy",
        "Anjeva",
        "Ankadifotsy",
        "Ankadikely Ilafy",
        "Ankadimbahoaka",
        "Ankadivato",
        "Ankazomanga",
        "Ankorondrano",
        "Anosibe",
        "Anosy",
        "Anosy Avaratra",
        "Antanimena",
        "Antsahabe",
        "Antsakaviro",
        "By-Pass",
        "Faravohitra",
        "Ilafy",
        "Isotry",
        "Itaosy",
        "Ivandry",
        "Mahamasina",
        "Mandroseza",
        "Namontana",
        "Sabotsy Namehana",
        "Soanierana",
        "Talatamaty",
        "Tanjombato",
        "Tsaralalàna",
    ],
)
superficie = st.number_input("Superficie (m²)", min_value=5, max_value=500, value=40)
nombre_chambres = st.number_input("Nombre de chambres", min_value=1, max_value=10, value=2)
douche_wc = st.radio("Douche / WC", ["interieur", "exterieur"])
type_d_acces = st.selectbox("Type d'accès", ["sans", "moto", "voiture", "voiture_avec_parking"])
meuble = st.radio("Meublé ?", ["oui", "non"])
etat_general = st.selectbox("État général", ["mauvais", "moyen", "bon"])

if st.button("Prédire le loyer"):
    # --- Reconstruction d'une ligne avec le MÊME prétraitement que dans le notebook ---
    ligne = pd.DataFrame([{
        "superficie": superficie,
        "nombre_chambres": nombre_chambres,
        "douche_wc": 1 if douche_wc == "interieur" else 0,
        "meuble": 1 if meuble == "oui" else 0,
        "etat_general": {"mauvais": 0, "moyen": 1, "bon": 2}[etat_general],
        "superficie_par_chambre": superficie / nombre_chambres,
    }])

    # One-hot encoding manuel pour quartier / type_d_acces (mêmes colonnes qu'à l'entraînement)
    for col in colonnes_modele:
        if col.startswith("quartier_"):
            ligne[col] = 1 if col == f"quartier_{quartier}" else 0
        elif col.startswith("type_d_acces_"):
            ligne[col] = 1 if col == f"type_d_acces_{type_d_acces}" else 0

    # --- Standardisation AVANT de restreindre aux colonnes du modèle ---
    # Important : si la RFE a éliminé "superficie", "nombre_chambres" ou
    # "superficie_par_chambre", ces colonnes n'existent plus dans `colonnes_modele`.
    # On ne peut donc standardiser que celles qui sont réellement présentes dans `ligne`.
    colonnes_a_standardiser_presentes = [c for c in colonnes_a_standardiser if c in ligne.columns]

    if colonnes_a_standardiser_presentes:
        # scaler a été entraîné (fit) sur la liste complète colonnes_a_standardiser, dans cet
        # ordre précis. Pour rester cohérent avec les moyennes/écarts-types appris, on ne peut
        # transformer que si on reconstruit exactement les mêmes colonnes, dans le même ordre.
        if len(colonnes_a_standardiser_presentes) == len(colonnes_a_standardiser):
            ligne[colonnes_a_standardiser] = scaler.transform(ligne[colonnes_a_standardiser])
        else:
            # Cas où certaines colonnes standardisées à l'entraînement ont été supprimées par
            # la RFE : on standardise quand même les colonnes présentes une par une, en
            # utilisant la moyenne/écart-type appris par le scaler pour chacune d'elles.
            index_col = {c: i for i, c in enumerate(colonnes_a_standardiser)}
            for col in colonnes_a_standardiser_presentes:
                i = index_col[col]
                ligne[col] = (ligne[col] - scaler.mean_[i]) / scaler.scale_[i]

    # S'assurer que toutes les colonnes attendues par le modèle existent, dans le bon ordre
    for col in colonnes_modele:
        if col not in ligne.columns:
            ligne[col] = 0
    ligne = ligne[colonnes_modele]

    # --- Prédiction ---
    prediction = model.predict(ligne)[0]

    # On stocke le résultat dans session_state : le composant carte (st_folium) déclenche
    # lui-même un rerun du script une fois chargé, ce qui remettrait st.button() à False et
    # ferait disparaître le résultat s'il n'était affiché que dans ce bloc "if".
    st.session_state["prediction"] = prediction
    st.session_state["coefs"] = pd.Series(model.coef_, index=colonnes_modele).sort_values(key=abs)

# --- Affichage du résultat (persiste tant qu'on ne relance pas une nouvelle prédiction) ---
if "prediction" in st.session_state:
    st.success(f"💰 Loyer mensuel estimé : {st.session_state['prediction']:,.0f} Ar")

    st.subheader("Poids des variables dans le modèle")
    fig, ax = plt.subplots()
    st.session_state["coefs"].plot(kind="barh", ax=ax)
    ax.set_xlabel("Poids (coefficient standardisé)")
    st.pyplot(fig)

# --- Bonus : carte interactive avec folium ---
st.subheader("📍 Carte (bonus)")
st.caption("Installer streamlit-folium (`pip install streamlit-folium folium`) pour activer cette section.")
try:
    import folium
    from streamlit_folium import st_folium

    coordonnees_quartiers = {
    "Analakely": (-18.9088, 47.5255),
    "Ankorondrano": (-18.8814, 47.5227),
    "Ivandry": (-18.8792, 47.5266),
    "Ambatobe": (-18.8663, 47.5450),
    "Isotry": (-18.9126, 47.5164),
    "67 Ha": (-18.9228, 47.5168),
    "Ambohijatovo": (-18.9105, 47.5282),
    "Ambohidahy": (-18.9033, 47.5290),
    "Ambohipo": (-18.9355, 47.5403),
    "Ambanidia": (-18.9158, 47.5384),
    "Ambatoroka": (-18.8976, 47.5329),
    "Ankadifotsy": (-18.8939, 47.5235),
    "Ankadivato": (-18.9012, 47.5198),
    "Anosy": (-18.9168, 47.5296),
    "Andavamamba": (-18.9218, 47.5214),
    "Andohalo": (-18.9144, 47.5366),
    "Andraharo": (-18.8755, 47.5078),
    "Andravoahangy": (-18.8972, 47.5207),
    "Ankazomanga": (-18.8920, 47.5018),
    "Anosy Avaratra": (-18.9107, 47.5308),
    "Antanimena": (-18.8929, 47.5177),
    "Antsahabe": (-18.8858, 47.5165),
    "Antsakaviro": (-18.8975, 47.5294),
    "Ankadimbahoaka": (-18.9224, 47.5275),
    "Behoririka": (-18.9035, 47.5218),
    "Faravohitra": (-18.9058, 47.5342),
    "Mahamasina": (-18.9186, 47.5310),
    "Soanierana": (-18.9395, 47.5268),
    "Tsaralalàna": (-18.9055, 47.5227),
    "Anosibe": (-18.9254, 47.5102),
    "Namontana": (-18.8865, 47.5006),
    "Ampandrana": (-18.9018, 47.5358),
    "Ampasampito": (-18.8840, 47.5456),
    "Ampefiloha": (-18.9175, 47.5232),
    "Tanjombato": (-18.9644, 47.5265),
    "Talatamaty": (-18.8358, 47.4798),
    "Itaosy": (-18.9482, 47.4485),
    "Alarobia": (-18.8738, 47.5288),
    "Mandroseza": (-18.9448, 47.5436),
    "Alakamisy Fenoarivo": (-18.9101, 47.5662),
    "Sabotsy Namehana": (-18.8206, 47.5668),
    "Anosy Avaratra": (-18.9107, 47.5308),
    "Ambodivona": (-18.8867, 47.5146),
    "Ampitatafika": (-18.9161, 47.4548),
    "By-Pass": (-18.9060, 47.5670),
    "Ankadikely Ilafy": (-18.8424, 47.5654),
    "Ilafy": (-18.8420, 47.5795),
    "Anjeva": (-18.8510, 47.6075),
    "Ambohimangakely": (-18.8722, 47.5960),
    "Ambohimanga": (-18.7607, 47.5628),
}
    lat, lon = coordonnees_quartiers[quartier]
    carte = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup=quartier).add_to(carte)
    st_folium(carte, width=700, height=400)
except ImportError:
    st.info("Module folium / streamlit-folium non installé.")
