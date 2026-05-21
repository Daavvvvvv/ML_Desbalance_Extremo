"""Streamlit demo para la exposicion: Desbalance extremo (focal loss + cost-sensitive).

Correr:
    streamlit run app.py

Carga los modelos entrenados por src/train_models.py y permite jugar con el
umbral de decision en tiempo real, viendo como cambian las metricas y el costo
en el test set (no visto durante entrenamiento ni tuning).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"


# ---------------------------------------------------------------- carga
class SimpleNN(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS_DIR / "meta.json").read_text())

    lr = joblib.load(MODELS_DIR / "lr_class_weight.joblib")

    nn_ckpt = torch.load(MODELS_DIR / "nn_focal_loss.pt", weights_only=False)
    nn_model = SimpleNN(nn_ckpt["in_dim"])
    nn_model.load_state_dict(nn_ckpt["state_dict"])
    nn_model.eval()
    scaler_mean = np.array(nn_ckpt["scaler_mean"])
    scaler_scale = np.array(nn_ckpt["scaler_scale"])

    test_df = pd.read_parquet(MODELS_DIR / "test_set.parquet")

    return meta, lr, nn_model, scaler_mean, scaler_scale, test_df


@st.cache_data
def compute_scores(_lr, _nn_model, scaler_mean, scaler_scale, test_df, feature_cols):
    X = test_df[feature_cols].values
    y = test_df["Class"].values

    score_lr = _lr.predict_proba(X)[:, 1]

    X_sc = (X - scaler_mean) / scaler_scale
    with torch.no_grad():
        logits = _nn_model(torch.tensor(X_sc, dtype=torch.float32))
        score_nn = torch.sigmoid(logits).numpy()

    return y, score_lr, score_nn


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Desbalance extremo - Demo", layout="wide")

st.title("Desbalance extremo: Cost-Sensitive vs Focal Loss")
st.caption(
    "Exposicion ML EAFIT 2026-1 — demo interactivo sobre el test set "
    "(transacciones no vistas durante entrenamiento ni tuning)"
)

if not (MODELS_DIR / "meta.json").exists():
    st.error(
        "No se encontraron modelos entrenados en `models/`. "
        "Corre primero: `python -m src.train_models`"
    )
    st.stop()

meta, lr, nn_model, scaler_mean, scaler_scale, test_df = load_artifacts()
y_test, score_lr, score_nn = compute_scores(
    lr, nn_model, scaler_mean, scaler_scale, test_df, meta["feature_cols"]
)

# ---------------- sidebar
st.sidebar.header("Configuracion")

model_name = st.sidebar.radio(
    "Modelo a evaluar",
    ["LR + class_weight='balanced'", "NN + Focal Loss (gamma=2)"],
)
score = score_lr if "LR" in model_name else score_nn
default_thr = (
    meta["thresholds_tuned_on_val"]["lr_class_weight"]
    if "LR" in model_name
    else meta["thresholds_tuned_on_val"]["nn_focal_loss"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Costos asimetricos**")
cost_fp = st.sidebar.number_input("Costo de un Falso Positivo ($)", value=1.0, min_value=0.0)
cost_fn = st.sidebar.number_input("Costo de un Falso Negativo ($)", value=100.0, min_value=0.0)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Umbral de decision",
    min_value=0.0, max_value=1.0,
    value=float(default_thr), step=0.001, format="%.3f",
)
st.sidebar.caption(f"Umbral tuneado en val (cost-min): {default_thr:.3f}")

# ---------------- main
y_pred = (score >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

pr_auc = average_precision_score(y_test, score)
roc_auc = roc_auc_score(y_test, score)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)
total_cost = fp * cost_fp + fn * cost_fn

st.subheader(f"Resultados sobre TEST — {model_name}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("PR-AUC (threshold-free)", f"{pr_auc:.4f}")
c2.metric("ROC-AUC", f"{roc_auc:.4f}")
c3.metric("Recall (a este umbral)", f"{recall:.3f}")
c4.metric("Precision (a este umbral)", f"{precision:.3f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("F1", f"{f1:.3f}")
c6.metric("True Positives", int(tp))
c7.metric("False Negatives (fraude no visto)", int(fn))
c8.metric(f"Costo total ($)", f"{total_cost:,.0f}")

# matriz de confusion
st.subheader("Matriz de confusion")
cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Real 0 (no fraude)", "Real 1 (fraude)"],
    columns=["Pred 0", "Pred 1"],
)
st.dataframe(cm_df, use_container_width=True)

# curva precision-recall
st.subheader("Curva Precision-Recall (sobre test)")
prec, rec, thrs = precision_recall_curve(y_test, score)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(rec, prec, linewidth=2, label=f"{model_name} (AP={pr_auc:.3f})")
# marcar el punto del umbral actual
y_pred_thr = (score >= threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_thr)
p_now = cm[1, 1] / max(cm[1, 1] + cm[0, 1], 1)
r_now = cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1)
ax.scatter([r_now], [p_now], color="red", s=100, zorder=5,
           label=f"umbral actual ({threshold:.3f})")
ax.axhline(y_test.mean(), color="gray", linestyle="--", alpha=0.5,
           label=f"baseline aleatorio ({y_test.mean():.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# inspeccion individual
st.subheader("Inspeccion individual de transacciones")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Sample fraude real"):
        st.session_state["sample_idx"] = int(
            test_df[test_df["Class"] == 1].sample(1).index[0]
        )
with col_b:
    if st.button("Sample no-fraude"):
        st.session_state["sample_idx"] = int(
            test_df[test_df["Class"] == 0].sample(1).index[0]
        )

if "sample_idx" in st.session_state:
    idx = st.session_state["sample_idx"]
    row = test_df.loc[idx]
    s = score[test_df.index.get_loc(idx)]
    decision = "FRAUDE" if s >= threshold else "no fraude"
    real = "FRAUDE" if int(row["Class"]) == 1 else "no fraude"

    st.write(f"**Real:** {real}    |    **Probabilidad modelo:** {s:.4f}    |    **Decision (thr={threshold:.3f}):** {decision}")
    st.write(f"Amount: `{row['Amount']:.2f}`  |  Time: `{row['Time']:.0f}s`")
    with st.expander("Ver features completas"):
        st.dataframe(row.to_frame("valor"), use_container_width=True)

st.markdown("---")
st.caption(
    "Modelos entrenados en train (60%), umbrales tuneados en val (20%), "
    "metricas reportadas sobre test (20%, sin tocar durante el desarrollo)."
)
