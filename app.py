import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="📊 Workflow d’Analyse Temporelle",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DESIGN UX PROFESSIONNEL (CSS) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #1E88E5 !important;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #0D47A1;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALISATION DE L'ÉTAT (SESSION STATE) ---
if "ts" not in st.session_state:
    st.session_state.ts = None
if "period" not in st.session_state:
    st.session_state.period = 12
if "model_result" not in st.session_state:
    st.session_state.model_result = None
if "model_type" not in st.session_state:
    st.session_state.model_type = "ARIMA"

# --- FONCTIONS UTILITAIRES ---
def check_stationarity(series):
    adf_test = adfuller(series.dropna())
    kpss_test = kpss(series.dropna(), regression='c')
    return {'adf_p': adf_test[1], 'kpss_p': kpss_test[1]}

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

# --- BARRE LATÉRALE & WORKFLOW ---
st.sidebar.title("📊 Workflow d’Analyse")

steps = [
    "1. Préparation des Données",
    "2. Exploration & Décomposition",
    "3. Stationnarité & Différenciation",
    "4. Modélisation & Ajustement",
    "5. Diagnostic des Résidus",
    "6. Prévisions Futures"
]

# Barre de progression en HAUT du menu
progress_map = {s: (i + 1) / len(steps) for i, s in enumerate(steps)}

# Navigation
step = st.sidebar.radio("Navigation", steps)

# Progress calculation
progress_value = progress_map[step]
progress_percent = int(progress_value * 100)

# Display
st.sidebar.markdown(f"### 📍 Progression : **{progress_percent}%**")
st.sidebar.progress(progress_value)

# --- 1️⃣ PRÉPARATION DES DONNÉES ---
if step == "1. Préparation des Données":
    st.title("📂 1. Préparation des Données")
    st.info("🎯 **Objectif** : Charger, nettoyer et structurer votre série temporelle.")
    
    uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            sep = st.radio("Séparateur CSV", [",", ";"], horizontal=True)
            df = pd.read_csv(uploaded_file, sep=sep)
        else:
            df = pd.read_excel(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Aperçu")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Configuration")
            date_col = st.selectbox("Colonne Date", df.columns)
            val_col = st.selectbox("Colonne Valeur", df.columns)
            freq = st.selectbox("Fréquence", ["D (Journalier)", "W (Hebdomadaire)", "M (Mensuel)", "Q (Trimestriel)", "Y (Annuel)"])
        
        if st.button("Valider la Série"):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col).set_index(date_col)
                ts = df[val_col].resample(freq.split(" ")[0]).mean().interpolate()
                st.session_state.ts = ts
                st.success("✅ Série temporelle prête !")
                st.line_chart(ts)
            except Exception as e:
                st.error(f"Erreur : {e}")

# --- 2️⃣ EXPLORATION & DÉCOMPOSITION ---
elif step == "2. Exploration & Décomposition":
    st.title("🔍 2. Exploration & Décomposition")
    if st.session_state.ts is None:
        st.warning("⚠️ Veuillez d'abord charger des données à l'étape 1.")
    else:
        ts = st.session_state.ts
        st.info("🎯 **Objectif** : Isoler la tendance et la saisonnalité pour comprendre la structure des données.")
        
        st.subheader("Série Temporelle Originale")
        fig_orig = go.Figure()
        fig_orig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Valeur', line=dict(color='#1E88E5')))
        fig_orig.update_layout(xaxis_title="Date", yaxis_title="Valeur", yaxis=dict(autorange=True, fixedrange=False))
        st.plotly_chart(fig_orig, use_container_width=True)
        
        st.divider()
        
        st.subheader("Décomposition de la Série")
        col_dec1, col_dec2 = st.columns([1, 3])
        with col_dec1:
            period_options = {"7 (Hebdomadaire)": 7, "12 (Mensuel)": 12, "4 (Trimestriel)": 4, "52 (Hebdomadaire)": 52, "365 (Annuel)": 365}
            selected_period = st.selectbox("Période saisonnière", list(period_options.keys()), index=1)
            st.session_state.period = period_options[selected_period]
            method = st.selectbox("Méthode", ["STL (Robuste)", "Additive", "Multiplicative"])
        
        with col_dec2:
            if method == "STL (Robuste)":
                decomp = STL(ts, period=st.session_state.period, robust=True).fit()
            else:
                decomp = seasonal_decompose(ts, model=method.lower(), period=st.session_state.period)
            
            fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            ts.plot(ax=axs[0], color='#1E88E5', title="Original")
            decomp.trend.plot(ax=axs[1], color='#E53935', title="Tendance")
            decomp.seasonal.plot(ax=axs[2], color='#43A047', title="Saisonnalité")
            decomp.resid.plot(ax=axs[3], color='#757575', style='.', title="Résidus")
            plt.tight_layout()
            st.pyplot(fig)

# --- 3️⃣ STATIONNARITÉ & DIFFÉRENCIATION ---

elif step == "3. Stationnarité & Différenciation":
    st.title("📈 3. Stationnarité & Différenciation")
    if st.session_state.ts is None:
        st.warning("⚠️ Veuillez d'abord charger des données.")
    else:
        ts = st.session_state.ts
        st.info("🎯 **Objectif** : Rendre la série stationnaire (moyenne et variance constantes) pour les modèles ARIMA/SARIMA.")
        
        orig = check_stationarity(ts)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ADF p-value", f"{orig['adf_p']:.4f}")
            st.write("✅ Stationnaire" if orig['adf_p'] < 0.05 else "❌ Non-stationnaire")
        with c2:
            st.metric("KPSS p-value", f"{orig['kpss_p']:.4f}")
            st.write("✅ Stationnaire" if orig['kpss_p'] > 0.05 else "❌ Non-stationnaire")
        
        st.divider()
        st.subheader("2. Différenciation")
        modele_choisi = st.radio("Pour quel modèle préparez-vous la série ?", ["ARIMA", "SARIMA"], horizontal=True)
        d = st.slider("Ordre de différenciation standard (d)", 0, 2, 0)
        D = 0
        if modele_choisi == "SARIMA":
            D = st.slider("Ordre de différenciation saisonnière (D)", 0, 2, 0)
        
        
        # Application des différenciations
        ts_final = ts.copy()
        if d > 0:
            for _ in range(d):
                ts_final = ts_final.diff().dropna()
        if D > 0:
            ts_final = ts_final.diff(st.session_state.period).dropna()
            
        if d > 0 or D > 0:
            new_results = check_stationarity(ts_final)
            st.write(f"**Résultats après différenciation (d={d}, D={D}) :**")
            res_col1, res_col2 = st.columns(2)
            res_col1.write(f"Nouvelle p-value ADF: **{new_results['adf_p']:.4f}**")
            res_col2.write(f"Nouvelle p-value KPSS: **{new_results['kpss_p']:.4f}**")
            if new_results['adf_p'] < 0.05 and new_results['kpss_p'] > 0.05:
                st.success("La série est maintenant stationnaire !")
            else:
                st.warning("La série n'est pas encore parfaitement stationnaire.")

        st.divider()
    
        st.subheader("3. Corrélogrammes (ACF & PACF)")
        st.write("Utilisez ces graphiques pour choisir les paramètres p, q (et P, Q pour SARIMA).")
        lags_auto = st.session_state.period * 5
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(ts_final, lags=lags_auto, ax=ax1)
        plot_pacf(ts_final, lags=lags_auto, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig)

# --- 4️⃣ MODÉLISATION & AJUSTEMENT ---
elif step == "4. Modélisation & Ajustement":
    st.title("🧠 4. Modélisation & Ajustement")
    if st.session_state.ts is None:
        st.warning("⚠️ Veuillez d'abord charger des données.")
    else:
        ts = st.session_state.ts
        st.info("🎯 **Objectif** : Entraîner le modèle sur les données historiques et valider sa performance.")
        
        test_size = st.slider("Taille du test (%)", 5, 30, 20) / 100
        split_idx = int(len(ts) * (1 - test_size))
        train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]
        
        model_type = st.selectbox("Choisir un modèle", ["ARIMA", "SARIMA"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Paramètres Non-Saisonniers (p, d, q)**")
            p = st.number_input("p", 0, 10, 1)
            d_val = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 10, 1)
        if model_type == "SARIMA":
            with col2:
                st.write("**Paramètres Saisonniers (P, D, Q, s)**")
                P = st.number_input("P", 0, 10, 0)
                D_val = st.number_input("D", 0, 1, 0)
                Q = st.number_input("Q", 0, 10, 0)
                s = st.number_input("s (Période)", value=st.session_state.period)
        
        if st.button(f"Entraîner {model_type}"):
            with st.spinner("Calcul en cours..."):
                if model_type == "ARIMA":
                    model = ARIMA(train, order=(p, d_val, q)).fit()
                else:
                    model = SARIMAX(train, order=(p, d_val, q), seasonal_order=(P, D_val, Q, s)).fit(disp=False)
                # 🔴 THIS LINE WAS MISSING
                st.session_state.model_result = model
                st.session_state.model_type = model_type

                
                preds = model.get_forecast(steps=len(test)).predicted_mean
                conf_int = model.get_forecast(steps=len(test)).conf_int()
                mae, rmse, mape = calculate_metrics(test, preds)
                
                st.subheader("Résultats de l'Évaluation")
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.2f}")
                c2.metric("RMSE", f"{rmse:.2f}")
                c3.metric("MAPE", f"{mape:.2%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train, name='Train'))
                fig.add_trace(go.Scatter(x=test.index, y=test, name='Test'))
                fig.add_trace(go.Scatter(x=test.index, y=preds, name='Prédictions', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=test.index, y=conf_int.iloc[:, 0], line_color='rgba(0,0,0,0)', showlegend=False))
                fig.add_trace(go.Scatter(x=test.index, y=conf_int.iloc[:, 1], fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line_color='rgba(0,0,0,0)', name='Confiance'))
                st.plotly_chart(fig, use_container_width=True)

# --- 5️⃣ DIAGNOSTIC DES RÉSIDUS ---
elif step == "5. Diagnostic des Résidus":
    st.title("🧪 5. Diagnostic des Résidus")
    if st.session_state.model_result is None:
        st.warning("⚠️ Veuillez d'abord ajuster un modèle à l'étape 4.")
    else:
        st.info("🎯 **Objectif** : Vérifier que les erreurs du modèle sont un 'bruit blanc' (pas d'information restante).")
        result = st.session_state.model_result
        resid = result.resid.dropna()
        
        # Tests en HAUT
        st.subheader("Tests Statistiques")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_p = lb_test['lb_pvalue'].iloc[0]
            st.metric("Ljung-Box p-value", f"{lb_p:.4f}")
            st.write("✅ Bruit blanc" if lb_p > 0.05 else "❌ Autocorrélation détectée")
        with col_r2:
            res_adf = adfuller(resid)[1]
            st.metric("ADF p-value (Résidus)", f"{res_adf:.4f}")
            st.write("✅ Stationnaires" if res_adf < 0.05 else "❌ Non-stationnaires")
            
        st.divider()
        
        # Graphiques en BAS
        st.subheader("Graphiques de Diagnostic")
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Résidus Standardisés
        axs[0, 0].plot(resid)
        axs[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axs[0, 0].set_title("Résidus Standardisés")
        axs[0, 0].tick_params(axis='x', rotation=45)

        
        # 2. Distribution Complexe (Histogramme + KDE + Normale)
        sns.histplot(resid, kde=True, ax=axs[0, 1], color='blue', stat="density", label="KDE")
        x_axis = np.linspace(resid.min(), resid.max(), 100)
        axs[0, 1].plot(x_axis, norm.pdf(x_axis, 0, 1), color='orange', label="N(0,1)")
        axs[0, 1].set_title("Distribution des Résidus")
        axs[0, 1].legend()
        
        # 3. Q-Q Plot
        qqplot(resid, line='s', ax=axs[1, 0])
        axs[1, 0].set_title("Normal Q-Q")
        
        # 4. Corrélogramme (ACF)
        plot_acf(resid, lags=20, ax=axs[1, 1])
        axs[1, 1].set_title("Corrélogramme (ACF)")
        
        plt.tight_layout()
        st.pyplot(fig)

# --- 6️⃣ PRÉVISIONS FUTURES ---
elif step == "6. Prévisions Futures":
    st.title("🔮 6. Prévisions Futures")

    if st.session_state.model_result is None:
        st.warning("⚠️ Veuillez d'abord ajuster un modèle.")
    else:
        st.info("🎯 **Objectif** : Projeter la série dans le futur avec des intervalles de confiance.")
        
        horizon = st.number_input("Horizon de prévision", 1, 100, 24)

        if st.button("Générer les Prévisions"):

            ts_full = st.session_state.ts
            result = st.session_state.model_result

            # 🔁 Refit model on FULL data
            if st.session_state.model_type == "ARIMA":
                final_model = ARIMA(
                    ts_full,
                    order=result.model.order
                ).fit()
            else:
                final_model = SARIMAX(
                    ts_full,
                    order=result.model.order,
                    seasonal_order=result.model.seasonal_order
                ).fit(disp=False)

            forecast = final_model.get_forecast(steps=horizon)
            y_pred = forecast.predicted_mean
            conf_int = forecast.conf_int()

            fig = go.Figure()

            # Historique
            fig.add_trace(go.Scatter(
                x=ts_full.index,
                y=ts_full,
                name="Historique"
            ))

            # Prévisions
            fig.add_trace(go.Scatter(
                x=y_pred.index,
                y=y_pred,
                name="Futur",
                line=dict(color="green", dash="dash")
            ))

            # Intervalle de confiance
            fig.add_trace(go.Scatter(
                x=y_pred.index,
                y=conf_int.iloc[:, 0],
                line_color="rgba(0,0,0,0)",
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=y_pred.index,
                y=conf_int.iloc[:, 1],
                fill="tonexty",
                fillcolor="rgba(0,255,0,0.1)",
                line_color="rgba(0,0,0,0)",
                name="Confiance"
            ))

            fig.update_layout(
                yaxis=dict(autorange=True, fixedrange=False),
                xaxis_title="Date",
                yaxis_title="Valeur"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "📥 Télécharger CSV",
                y_pred.to_csv(),
                "forecast.csv"
            )
