
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration page ─────────────────────────────────────
st.set_page_config(
    page_title="AquaCheck — Potabilité de l'eau",
    page_icon="💧",
    layout="wide"
)

# ── Chargement du modèle ───────────────────────────────────
@st.cache_resource
def load_model():
    with open('water_model_package.pkl', 'rb') as f:
        return pickle.load(f)

pkg = load_model()
model    = pkg['model']
imputer  = pkg['imputer']
scaler   = pkg['scaler']
seuil    = pkg['seuil']
who      = pkg['who_limits']
features_originales = pkg['features_originales']
features_finales    = pkg['features_finales']
metrics  = pkg['metrics']

# ── Style CSS custom ───────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a1628; }
    .block-container { padding: 2rem 3rem; }
    h1 { color: #ffffff !important; font-size: 2rem !important; }
    h3 { color: #cbd5e1 !important; }
    .stSlider label { color: #94a3b8 !important; font-size: 0.85rem !important; }
    .metric-card {
        background: #0d1f3c;
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 600; color: #fff; }
    .metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .verdict-potable {
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-danger {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .alert-box {
        background: rgba(239,68,68,0.06);
        border: 0.5px solid rgba(239,68,68,0.25);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .ok-box {
        background: rgba(34,197,94,0.06);
        border: 0.5px solid rgba(34,197,94,0.2);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    div[data-testid="stMarkdownContainer"] p { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("# 💧 AquaCheck")
st.markdown("**Outil de prédiction de potabilité de l'eau** · Modèle Stacking (XGBoost + Random Forest + LR)")

# Métriques du modèle en haut
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{metrics['auc_roc']}</div>
        <div class="metric-lbl">AUC-ROC</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{round(metrics['recall']*100, 1)}%</div>
        <div class="metric-lbl">Recall</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{round(metrics['fn']/(metrics['fn']+256)*100, 1)}%</div>
        <div class="metric-lbl">Taux erreur critique</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{seuil}</div>
        <div class="metric-lbl">Seuil décision</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL : saisie gauche | résultats droite
# ══════════════════════════════════════════════════════════
col_input, col_result = st.columns([1, 1], gap="large")

# ── COLONNE GAUCHE : sliders ───────────────────────────────
with col_input:
    st.markdown("### Paramètres mesurés")

    params = {
        'ph':              st.slider('pH',                    0.0,  14.0,  7.0,  0.1),
        'Hardness':        st.slider('Hardness (mg/L)',       47.0, 323.0, 196.0, 1.0),
        'Solids':          st.slider('Solids (mg/L)',         320.0, 61227.0, 22000.0, 10.0),
        'Chloramines':     st.slider('Chloramines (mg/L)',    0.0,  13.0,  7.0,  0.1),
        'Sulfate':         st.slider('Sulfate (mg/L)',        129.0, 481.0, 333.0, 1.0),
        'Conductivity':    st.slider('Conductivity (µS/cm)', 181.0, 753.0, 426.0, 1.0),
        'Organic_carbon':  st.slider('Organic carbon (mg/L)', 2.0, 28.0,  14.0,  0.1),
        'Trihalomethanes': st.slider('Trihalomethanes (µg/L)',0.0, 124.0,  66.0,  0.1),
        'Turbidity':       st.slider('Turbidity (NTU)',       1.0,   6.7,   4.0,  0.1),
    }

    analyser = st.button("🔍 Analyser cette eau", use_container_width=True)

# ── COLONNE DROITE : résultats ─────────────────────────────
with col_result:
    st.markdown("### Résultat de l'analyse")

    if analyser:

        # ── Feature engineering (identique à l'entraînement) ──
        ph = params['ph']
        row = {**params}
        row['ph_optimal']    = float((ph >= 6.5) and (ph <= 8.5))
        row['ph_hors_norme'] = float((ph < 6.5) or (ph > 8.5))
        row['ph_extreme']    = float((ph < 5.0) or (ph > 10.0))
        row['ph_alcalin']    = float(ph > 8.5)
        row['score_risque']  = (
            float((ph < 6.5) or (ph > 8.5)) +
            float(params['Turbidity'] > 4.0) +
            float(params['Chloramines'] > 4.0) +
            float(params['Sulfate'] > 250) +
            float(params['Trihalomethanes'] > 80)
        )
        row['ph_x_solids'] = ph * params['Solids']

        # ── Preprocessing ──────────────────────────────────────
        X_input = pd.DataFrame([row])[features_finales]
        X_imp   = imputer.transform(X_input)
        X_sc    = scaler.transform(X_imp)

        # ── Prédiction ─────────────────────────────────────────
        proba    = model.predict_proba(X_sc)[0][1]
        verdict  = proba >= seuil
        pct      = round(proba * 100, 1)

        # ── Affichage verdict ──────────────────────────────────
        if verdict:
            st.markdown(f"""<div class="verdict-potable">
                <div style="font-size:2.5rem">✅</div>
                <div style="font-size:1.6rem; font-weight:600; color:#4ade80; margin:8px 0">POTABLE</div>
                <div style="color:#94a3b8; font-size:0.9rem">Probabilité : {pct}% &gt; seuil {int(seuil*100)}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="verdict-danger">
                <div style="font-size:2.5rem">⛔</div>
                <div style="font-size:1.6rem; font-weight:600; color:#f87171; margin:8px 0">NON POTABLE</div>
                <div style="color:#94a3b8; font-size:0.9rem">Probabilité : {pct}% &lt; seuil {int(seuil*100)}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Paramètres hors norme OMS ──────────────────────────
        st.markdown("**Contrôle normes OMS**")
        alertes = []
        for feat, (lo, hi) in who.items():
            val = params[feat]
            hors = (lo is not None and val < lo) or (hi is not None and val > hi)
            norme_str = f"{lo}–{hi}" if lo else f"< {hi}"
            if hors:
                alertes.append((feat, val, norme_str))
                st.markdown(f"""<div class="alert-box">
                    ⚠️ <b style="color:#f87171">{feat}</b> = {val}
                    &nbsp;·&nbsp; <span style="color:#94a3b8">norme OMS : {norme_str}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="ok-box">
                    ✓ <b style="color:#4ade80">{feat}</b> = {val}
                    &nbsp;·&nbsp; <span style="color:#64748b">norme OMS : {norme_str}</span>
                </div>""", unsafe_allow_html=True)

        # ── SHAP ───────────────────────────────────────────────
        st.markdown("<br>**Pourquoi ce verdict ? (SHAP)**", unsafe_allow_html=True)

        try:
            # Utiliser le XGBoost du stacking pour SHAP
            xgb_estimator = model.named_estimators_['xgb']
            explainer   = shap.TreeExplainer(xgb_estimator)
            shap_vals   = explainer.shap_values(X_sc)

            shap_df = pd.DataFrame({
                'feature': features_finales,
                'shap':    shap_vals[0]
            }).reindex(pd.Series(np.abs(shap_vals[0])).sort_values(ascending=False).index)
            shap_df = shap_df.head(8)

            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#0d1f3c')
            ax.set_facecolor('#0d1f3c')

            colors = ['#4ade80' if v > 0 else '#f87171' for v in shap_df['shap']]
            bars = ax.barh(shap_df['feature'], shap_df['shap'], color=colors, height=0.6)
            ax.axvline(0, color='rgba(255,255,255,0.2)', linewidth=0.8)
            ax.set_xlabel('Impact SHAP', color='#64748b', fontsize=9)
            ax.tick_params(colors='#94a3b8', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a5f')
            ax.invert_yaxis()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info(f"SHAP non disponible : {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#334155;">
            <div style="font-size:3rem">💧</div>
            <div style="margin-top:1rem; font-size:1rem">
                Ajuste les paramètres et clique sur Analyser
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#334155; font-size:0.8rem'>"
    "Projet H2 · Stacking (XGBoost + RF + LR) · "
    "AUC 0.674 · Seuil 0.30 · Dataset : Water Potability (Kaggle)"
    "</div>",
    unsafe_allow_html=True
)
