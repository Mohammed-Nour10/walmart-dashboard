"""
Tableau de bord predictif — Prevision de la demande Walmart
Projet de recherche — Centrale Lille — M.-N. Attoubi
Encadrant : Pr. Ahmed Rahmani (CRIStAL)

Lancement :
    pip install streamlit          # si pas deja installe
    streamlit run dashboard.py     # depuis la racine du projet

Le fichier doit etre place a la racine du projet, a cote du dossier data/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────── Config ───────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
DATE_SPLIT = pd.Timestamp("2012-06-01")        # meme coupure que les notebooks

st.set_page_config(
    page_title="Prevision demande Walmart",
    page_icon="📦",
    layout="wide",
)

# Resultats du benchmark (notebooks 03 a 06, serie de reference Store 1 / Dept 1)
BENCHMARK = pd.DataFrame({
    "Modele": ["Naive", "ARIMA", "SARIMA", "Prophet", "XGBoost", "LSTM"],
    "MAE":    [2075.5, 4599.8, 1264.0, 1260.7,  848.8, 2090.5],
    "RMSE":   [3081.7, 4992.1, 2027.7, 1746.6, 1022.4, 2602.2],
    "MAPE":   [ 10.14,  26.59,   6.33,   6.31,   4.77,  11.66],
})


# ─────────────────────────────── Donnees ──────────────────────────────
@st.cache_data
def charger_donnees():
    """Charge le dataset nettoye produit par 02_preprocessing."""
    df = pd.read_csv(DATA_DIR / "df_clean.csv", parse_dates=["Date"])
    return df


def creer_features(df):
    """Feature engineering identique au notebook 05_XGBoost."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Lags
    for lag in [1, 2, 3, 4, 8, 12, 52]:
        df[f"lag_{lag}"] = df["Weekly_Sales"].shift(lag)

    # Statistiques glissantes (shift(1) -> pas de fuite)
    for w in [4, 8, 12]:
        df[f"rolling_mean_{w}"] = df["Weekly_Sales"].shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"]  = df["Weekly_Sales"].shift(1).rolling(w).std()
        df[f"rolling_max_{w}"]  = df["Weekly_Sales"].shift(1).rolling(w).max()

    # Variables calendaires
    df["semaine"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["mois"]      = df["Date"].dt.month
    df["trimestre"] = df["Date"].dt.quarter
    df["annee"]     = df["Date"].dt.year

    # Encodage cyclique
    df["sem_sin"]  = np.sin(2 * np.pi * df["semaine"] / 52)
    df["sem_cos"]  = np.cos(2 * np.pi * df["semaine"] / 52)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

    # Indicateurs binaires
    df["est_Q4"]  = (df["trimestre"] == 4).astype(int)
    df["est_ete"] = df["mois"].isin([6, 7, 8]).astype(int)
    if "IsHoliday" in df.columns:
        df["est_ferie"] = df["IsHoliday"].astype(int)

    # Differences
    df["diff_1"]  = df["Weekly_Sales"].shift(1).diff(1)
    df["diff_52"] = df["Weekly_Sales"].shift(1).diff(52)

    return df.dropna().reset_index(drop=True)


@st.cache_resource
def entrainer_modele(store, dept):
    """Entraine XGBoost sur la serie (store, dept). Resultat mis en cache."""
    df = charger_donnees()
    serie = df[(df["Store"] == store) & (df["Dept"] == dept)].copy()

    if len(serie) < 70:                      # besoin d'~1 an d'historique (lag_52)
        return None

    feat  = creer_features(serie)
    train = feat[feat["Date"] <  DATE_SPLIT]
    test  = feat[feat["Date"] >= DATE_SPLIT]

    if len(train) < 20 or len(test) < 4:     # serie trop courte apres decoupage
        return None

    exclure = ["Date", "Weekly_Sales", "IsHoliday", "Store", "Dept", "Type", "Size"]
    cols = [c for c in feat.columns
            if c not in exclure and pd.api.types.is_numeric_dtype(feat[c])]

    # Validation tail dans le train pour l'early stopping (jamais le test)
    n_val = min(13, max(4, len(train) // 5))
    X_tr,  y_tr  = train[cols].iloc[:-n_val], train["Weekly_Sales"].iloc[:-n_val]
    X_val, y_val = train[cols].iloc[-n_val:], train["Weekly_Sales"].iloc[-n_val:]

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        early_stopping_rounds=30, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(test[cols])

    return {"model": model, "cols": cols, "feat": feat,
            "train": train, "test": test, "pred": pred}


def calculer_metriques(y_reel, y_pred):
    mae  = mean_absolute_error(y_reel, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reel, y_pred))
    mask = y_reel != 0
    mape = np.mean(np.abs((y_reel[mask] - y_pred[mask]) / y_reel[mask])) * 100
    return mae, rmse, mape


# ─────────────────────────────── Interface ────────────────────────────
st.title("📦 Tableau de bord predictif — Prevision de la demande")
st.caption("Walmart Store Sales · modele XGBoost · projet de recherche Centrale Lille")

df = charger_donnees()

# Sidebar — selection de la serie
st.sidebar.header("Parametres")
stores = sorted(df["Store"].unique())
store = st.sidebar.selectbox(
    "Magasin (Store)", stores,
    index=stores.index(1) if 1 in stores else 0)
depts = sorted(df[df["Store"] == store]["Dept"].unique())
dept = st.sidebar.selectbox(
    "Rayon (Dept)", depts,
    index=depts.index(1) if 1 in depts else 0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Serie analysee**\n\n"
    f"Store {store} · Dept {dept}\n\n"
    f"Coupure train/test : {DATE_SPLIT.date()}")

res = entrainer_modele(store, dept)

tab1, tab2, tab3 = st.tabs(["📈 Prevision", "🏆 Benchmark", "🔍 Variables"])

# ── Onglet 1 : Prevision ──
with tab1:
    if res is None:
        st.warning(
            "Serie trop courte pour entrainer le modele "
            "(il faut environ un an d'historique). "
            "Choisis un autre couple Store / Dept dans le menu de gauche.")
    else:
        y_test = res["test"]["Weekly_Sales"].values
        pred   = res["pred"]
        mae, rmse, mape = calculer_metriques(y_test, pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"{mae:,.0f} $")
        c2.metric("RMSE", f"{rmse:,.0f} $")
        c3.metric("MAPE", f"{mape:.2f} %")

        import matplotlib.pyplot as plt

        train, test = res["train"], res["test"]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(train["Date"], train["Weekly_Sales"],
                color="#0D47A1", lw=1.2, label="Historique (train)")
        ax.plot(test["Date"], test["Weekly_Sales"],
                color="#2E7D32", lw=1.6, label="Réel (test)")
        ax.plot(test["Date"], pred,
                color="#E65100", lw=1.6, ls="--", label="Prévision XGBoost")
        ax.axvline(test["Date"].iloc[0], color="red", ls=":", alpha=0.7)
        ax.set_title(f"Store {store} / Dept {dept}", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            f"Bleu = historique d'entrainement · vert = ventes reelles du test · "
            f"orange (pointillés) = prevision XGBoost. Test a partir du "
            f"{DATE_SPLIT.date()} ({len(y_test)} semaines jamais vues a l'entrainement).")

# ── Onglet 2 : Benchmark ──
with tab2:
    st.subheader("Comparaison des 6 modeles — serie de reference (Store 1 / Dept 1)")
    st.dataframe(
        BENCHMARK.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE"], color="#C8E6C9"),
        use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Erreurs absolues ($)**")
        st.bar_chart(BENCHMARK.set_index("Modele")[["MAE", "RMSE"]])
    with col_b:
        st.markdown("**Erreur relative (%)**")
        st.bar_chart(BENCHMARK.set_index("Modele")[["MAPE"]])

    st.info(
        "XGBoost domine sur les trois metriques (MAPE 4,77 %). "
        "Le LSTM, sous-alimente en donnees (~50 semaines d'entrainement), "
        "reste au niveau du modele naif : la sophistication d'un modele "
        "ne garantit rien quand la donnee est rare.")

# ── Onglet 3 : Variables ──
with tab3:
    if res is None:
        st.warning("Selectionne une serie valide pour afficher l'importance des variables.")
    else:
        imp = (pd.DataFrame({"feature": res["cols"],
                             "importance": res["model"].feature_importances_})
               .sort_values("importance", ascending=False)
               .head(15)
               .set_index("feature"))
        st.subheader("Top 15 des variables les plus importantes")
        st.bar_chart(imp)
        st.caption(
            "Plus la barre est longue, plus la variable pese dans les "
            "predictions. Les lags (notamment lag_52, la valeur d'il y a un an) "
            "et les variables exogenes captent la saisonnalite et les promotions.")