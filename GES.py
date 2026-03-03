import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Simulateur Bourse GES", layout="wide")
# --- AFFICHAGE DES LOGOS ---
col_logo1, col_logo2, col_logo3 = st.columns(3)

with col_logo1:
    # Affiche le logo UMP (assurez-vous que le nom du fichier est exact)
    st.image("ump.png.png", width=320) 

with col_logo2:
    st.image("encg.png.png", width=250)

with col_logo3:
    st.image("facg.png.png", width=150)

st.markdown("---") # Ajoute une belle ligne de séparation sous les logos
st.title("📊 Prototype GES : Prédiction IA & Analyse Technique")
st.markdown("Ce simulateur lit vos indicateurs en temps réel depuis Google Sheets, génère une prédiction IA, et simule le comportement boursier du titre **Green Energy Solutions (GES)**.")

# LECTURE DEPUIS VOTRE GOOGLE SHEET
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ewwuOPVe4helz2wug0MsPzVd2k_uSbLEJzCPrFD0Naw/export?format=csv&gid=2024535575"

try:
    df_params = pd.read_csv(SHEET_URL)
    df_params.columns = df_params.columns.str.strip()
    
    prix_initial = float(df_params.loc[df_params['Paramètre'] == 'Prix_IPO', 'Valeur'].values[0])
    tendance_annuelle = float(df_params.loc[df_params['Paramètre'] == 'Croissance_Annuelle', 'Valeur'].values[0])
    volatilite = float(df_params.loc[df_params['Paramètre'] == 'Volatilite', 'Valeur'].values[0])
    annees = int(float(df_params.loc[df_params['Paramètre'] == 'Annees_Simulation', 'Valeur'].values[0]))
    
    st.sidebar.success("✅ Connecté au Google Sheet !")
    st.sidebar.dataframe(df_params)
except Exception as e:
    st.error("❌ Erreur de connexion au Google Sheet. Vérifiez que les valeurs sont bien des nombres (utilisez des points, pas des virgules).")
    st.stop()

# SIMULATION DU MARCHE ET OHLC (Open, High, Low, Close)
jours_par_an = 252
jours_cotation = annees * jours_par_an
tendance_journaliere = tendance_annuelle / jours_par_an 

np.random.seed(42)
rendements = np.random.normal(loc=tendance_journaliere, scale=volatilite, size=jours_cotation)
facteurs_prix = np.cumprod(1 + rendements)
prix_cloture = prix_initial * facteurs_prix

dates = pd.date_range(start="2025-07-01", periods=jours_cotation, freq='B')
df = pd.DataFrame({'Date': dates, 'Close': prix_cloture})
df['Jour_Index'] = np.arange(len(df))

# Création des données pour les chandeliers (bougies)
df['Open'] = df['Close'].shift(1).fillna(prix_initial)
# Ajout d'une volatilité intra-journalière pour créer les "mèches" des bougies
amplitude_jour = df['Close'] * (volatilite / np.sqrt(252)) * np.random.uniform(0.2, 1.0, size=jours_cotation)
df['High'] = df[['Open', 'Close']].max(axis=1) + amplitude_jour
df['Low'] = df[['Open', 'Close']].min(axis=1) - amplitude_jour

# INTELLIGENCE ARTIFICIELLE (Prédiction)
modele_ia = LinearRegression()
X = df[['Jour_Index']]
y = df['Close']
modele_ia.fit(X, y)
df['Prediction_IA'] = modele_ia.predict(X)

# AFFICHAGE DES DEUX GRAPHIQUES CÔTE À CÔTE
col_gauche, col_droite = st.columns(2)

# --- GRAPHIQUE 1 : TENDANCE IA (À Gauche) ---
with col_gauche:
    st.subheader(f"📈 Tendance Globale IA ({annees} ans)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label="Cours simulé", color='#1f77b4', alpha=0.5)
    ax.plot(df['Date'], df['Prediction_IA'], label="Tendance IA", color='#d62728', linewidth=3)
    ax.axhline(y=prix_initial, color='#2ca02c', linestyle='--', label=f"Prix d'IPO ({prix_initial} MAD)")
    ax.set_xlabel("Année")
    ax.set_ylabel("Prix de l'action (MAD)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# --- GRAPHIQUE 2 : CHANDELIERS JAPONAIS (À Droite) ---
with col_droite:
    st.subheader("🕯️ Analyse Technique (Chandeliers)")
    
    # Boutons pour choisir la durée
    choix_duree = st.radio(
        "Sélectionnez la fenêtre d'observation :", 
        ["1 Mois", "3 Mois", "6 Mois", "1 An", "Maximum"], 
        horizontal=True
    )
    
    # Filtrer les données selon le choix
    if choix_duree == "1 Mois":
        df_filtre = df.head(21) # 21 jours de bourse par mois
    elif choix_duree == "3 Mois":
        df_filtre = df.head(63)
    elif choix_duree == "6 Mois":
        df_filtre = df.head(126)
    elif choix_duree == "1 An":
        df_filtre = df.head(252)
    else:
        df_filtre = df

    # Création du graphique en chandeliers avec Plotly
    fig_candle = go.Figure(data=[go.Candlestick(x=df_filtre['Date'],
                    open=df_filtre['Open'],
                    high=df_filtre['High'],
                    low=df_filtre['Low'],
                    close=df_filtre['Close'],
                    increasing_line_color='green', decreasing_line_color='red')])
    
    fig_candle.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_rangeslider_visible=False,
        yaxis_title="Prix (MAD)",
        height=400
    )
    st.plotly_chart(fig_candle, use_container_width=True)

# METRIQUES FINALES
st.markdown("---")
prix_final = df['Close'].iloc[-1]
rendement_global = ((prix_final / prix_initial) - 1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Prix initial (IPO)", f"{prix_initial:.2f} MAD")
col2.metric(f"Prix estimé en fin de période", f"{prix_final:.2f} MAD", f"{rendement_global:.2f} %")
col3.metric("Volatilité simulée", f"{volatilite*100:.1f} %")






