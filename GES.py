import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulateur Bourse GES", layout="wide")

st.title("📊 Prototype GES : Prédiction Boursière par IA (5 ans)")
st.markdown("Ce simulateur lit vos indicateurs en temps réel depuis Google Sheets et génère une prédiction sur l'évolution du titre **Green Energy Solutions (GES)**.")

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
    st.error("❌ Erreur de connexion au Google Sheet.")
    st.stop()

# SIMULATION DU MARCHE
jours_par_an = 252
jours_cotation = annees * jours_par_an
tendance_journaliere = tendance_annuelle / jours_par_an 

np.random.seed(42)
rendements = np.random.normal(loc=tendance_journaliere, scale=volatilite, size=jours_cotation)
facteurs_prix = np.cumprod(1 + rendements)
prix_cloture = prix_initial * facteurs_prix

dates = pd.date_range(start="2025-07-01", periods=jours_cotation, freq='B')
df = pd.DataFrame({'Date': dates, 'Prix_GES': prix_cloture})
df['Jour_Index'] = np.arange(len(df))

# INTELLIGENCE ARTIFICIELLE
modele_ia = LinearRegression()
X = df[['Jour_Index']]
y = df['Prix_GES']
modele_ia.fit(X, y)
df['Prediction_IA'] = modele_ia.predict(X)

# GRAPHIQUE
st.subheader(f"📈 Simulation et Prédiction IA sur {annees} ans")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Date'], df['Prix_GES'], label="Cours simulé", color='#1f77b4', alpha=0.7)
ax.plot(df['Date'], df['Prediction_IA'], label="Tendance IA", color='#d62728', linewidth=3)
ax.axhline(y=prix_initial, color='#2ca02c', linestyle='--', label=f"Prix d'IPO ({prix_initial} MAD)")

ax.set_title("Évolution post-IPO de Green Energy Solutions", fontsize=14)
ax.set_xlabel("Année")
ax.set_ylabel("Prix de l'action (MAD)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# METRIQUES FINALES
prix_final = df['Prix_GES'].iloc[-1]
rendement_global = ((prix_final / prix_initial) - 1) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Prix IPO", f"{prix_initial:.2f} MAD")
col2.metric(f"Prix estimé ({2025+annees})", f"{prix_final:.2f} MAD", f"{rendement_global:.2f} %")
col3.metric("Croissance Annuelle", f"{tendance_annuelle*100:.1f} %")