import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Simulateur Bourse GES", layout="wide")

# --- AFFICHAGE DES LOGOS ---
col_logo1, col_logo2, col_logo3 = st.columns(3)
with col_logo1:
    st.image("ump.png.png", width=400) 
with col_logo2:
    st.image("encg.png.png", width=270)
with col_logo3:
    st.image("facg.png.png", width=130)

st.markdown("---")

# --- DATE, HEURE ET LIEU (OUJDA) ---
tz_maroc = pytz.timezone('Africa/Casablanca')
maintenant = datetime.now(tz_maroc)

col_entete1, col_entete2 = st.columns([3, 1])
with col_entete1:
    st.title("📊 Terminal GES : Prédiction IA & Live Trading")
with col_entete2:
    st.info(f"📍 **Oujda, Maroc** \n📅 {maintenant.strftime('%d/%m/%Y')}  \n⏰ **{maintenant.strftime('%H:%M:%S')}**")

st.markdown("---")

# --- LECTURE GOOGLE SHEETS & BARRE LATÉRALE ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/1ewwuOPVe4helz2wug0MsPzVd2k_uSbLEJzCPrFD0Naw/export?format=csv&gid=2024535575"

try:
    df_params = pd.read_csv(SHEET_URL)
    df_params.columns = df_params.columns.str.strip()
    
    prix_initial = float(df_params.loc[df_params['Paramètre'] == 'Prix_IPO', 'Valeur'].values[0])
    tendance_annuelle = float(df_params.loc[df_params['Paramètre'] == 'Croissance_Annuelle', 'Valeur'].values[0])
    volatilite = float(df_params.loc[df_params['Paramètre'] == 'Volatilite', 'Valeur'].values[0])
    annees = int(float(df_params.loc[df_params['Paramètre'] == 'Annees_Simulation', 'Valeur'].values[0]))
    
    # TABLEAU SUR LE CÔTÉ (SIDEBAR)
    st.sidebar.success("✅ Connecté au Google Sheet !")
    st.sidebar.dataframe(df_params)
    
except Exception as e:
    st.error("❌ Erreur Google Sheet. Vérifiez les valeurs.")
    st.stop()

# --- CREATION DES DEUX PAGES (ONGLETS) ---
onglet_ia, onglet_live = st.tabs(["📈 Prédiction IA (Long Terme)", "⚡ Analyse Chartiste (Temps Réel)"])

# ==========================================
# ONGLET 1 : L'INTELLIGENCE ARTIFICIELLE
# ==========================================
with onglet_ia:
    st.markdown(f"### Modèle Prédictif sur {annees} ans")
    jours_cotation = annees * 252
    tendance_journaliere = tendance_annuelle / 252 

    np.random.seed(42)
    rendements = np.random.normal(loc=tendance_journaliere, scale=volatilite, size=jours_cotation)
    prix_cloture = prix_initial * np.cumprod(1 + rendements)

    dates = pd.date_range(start=maintenant, periods=jours_cotation, freq='B')
    df_ia = pd.DataFrame({'Date': dates, 'Close': prix_cloture})
    df_ia['Jour_Index'] = np.arange(len(df_ia))

    modele_ia = LinearRegression()
    modele_ia.fit(df_ia[['Jour_Index']], df_ia['Close'])
    df_ia['Prediction_IA'] = modele_ia.predict(df_ia[['Jour_Index']])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_ia['Date'], df_ia['Close'], label="Cours simulé", color='#1f77b4', alpha=0.5)
    ax.plot(df_ia['Date'], df_ia['Prediction_IA'], label="Tendance IA", color='#d62728', linewidth=3)
    ax.axhline(y=prix_initial, color='#2ca02c', linestyle='--', label=f"Prix d'IPO ({prix_initial} MAD)")
    ax.set_ylabel("Prix de l'action (MAD)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    # RESULTATS EN DESSOUS DU GRAPHIQUE
    st.markdown("---")
    prix_final = df_ia['Close'].iloc[-1]
    rendement_global = ((prix_final / prix_initial) - 1) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Prix initial (IPO)", f"{prix_initial:.2f} MAD")
    col2.metric(f"Prix estimé à {annees} ans", f"{prix_final:.2f} MAD", f"{rendement_global:.2f} %")
    col3.metric("Volatilité simulée", f"{volatilite*100:.1f} %")

# ==========================================
# ONGLET 2 : LE TEMPS REEL (INTRADAY)
# ==========================================
with onglet_live:
    col_gauche, col_droite = st.columns([1, 4])
    
    with col_gauche:
        st.markdown("### Contrôle Live")
        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.success("Cotations actualisées à la seconde !")
            
        unite_temps = st.radio(
            "⏳ Unité de temps :",
            ["5 Minutes", "15 Minutes", "30 Minutes", "1 Heure"]
        )
        
        st.markdown("---")
        # NOUVEAU : Le curseur pour choisir le nombre de jours d'historique
        historique_jours = st.slider(
            "📅 Historique (en jours) :",
            min_value=1,
            max_value=30,
            value=5, # Valeur par défaut : 5 jours
            step=1
        )

    with col_droite:
        # On utilise maintenant la variable 'historique_jours' choisie par l'utilisateur
        date_debut_live = maintenant - timedelta(days=historique_jours)
        dates_live = pd.date_range(start=date_debut_live, end=maintenant, freq='1min')
        
        volatilite_min = volatilite / np.sqrt(252 * 6.5 * 60)
        rendements_min = np.random.normal(loc=0, scale=volatilite_min, size=len(dates_live))
        prix_live = prix_initial * np.cumprod(1 + rendements_min)
        
        df_live = pd.DataFrame({'Date': dates_live, 'Price': prix_live})
        df_live.set_index('Date', inplace=True)
        
        dict_resample = {
            "5 Minutes": "5min", "15 Minutes": "15min", 
            "30 Minutes": "30min", "1 Heure": "1h"
        }
        
        df_ohlc = df_live['Price'].resample(dict_resample[unite_temps]).ohlc()
        df_ohlc = df_ohlc.dropna() 

        dernier_prix = df_ohlc['close'].iloc[-1]
        variation = dernier_prix - prix_initial
        st.metric(label="Cotation GES en Direct (MAD)", value=f"{dernier_prix:.2f}", delta=f"{variation:.2f} MAD depuis l'IPO")

        # Affichage propre du graphique boursier
        fig_live = go.Figure(data=[go.Candlestick(
            x=df_ohlc.index,
            open=df_ohlc['open'], high=df_ohlc['high'],
            low=df_ohlc['low'], close=df_ohlc['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
        )])
        
        fig_live.update_layout(
            title=f"Analyse Technique Intraday - Bougies de {unite_temps}",
            yaxis_title="Prix (MAD)",
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_live, use_container_width=True)





