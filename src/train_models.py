"""Entrena los modelos finales y los guarda en models/ para que app.py los use.

Correr una sola vez:  python src/train_models.py

Reproduce el flujo del notebook (split temporal, tuning de umbral en val) pero
sin ploteos ni explicaciones. Salida: 3 archivos en models/:
  - lr_class_weight.joblib
  - nn_focal_loss.pt
  - meta.json  (con umbrales tuneados en val y feature columns)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .focal_loss import FocalLossBinary
from .metrics import cost_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def temporal_split(df: pd.DataFrame, train_frac=0.60, val_frac=0.20):
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    train = df.iloc[:n_tr]
    val = df.iloc[n_tr:n_tr + n_va]
    test = df.iloc[n_tr + n_va:]
    return train, val, test


def tune_threshold_for_cost(y_val, score_val, cost_fp=1.0, cost_fn=100.0):
    thresholds = np.linspace(0.001, 0.999, 200)
    costs = [
        cost_score(y_val, (score_val >= t).astype(int), cost_fp, cost_fn)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmin(costs))])


class SimpleNN(nn.Module):
    """Red simple para clasificacion binaria, identica a la del notebook."""

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


def train_nn(X_tr, y_tr, loss_fn, epochs=20, lr=1e-3, seed=42, device="cpu"):
    torch.manual_seed(seed)
    model = SimpleNN(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=2048, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def main():
    print(f"Cargando dataset desde {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}, positivos: {df.Class.sum()} ({df.Class.mean()*100:.3f}%)")

    train_df, val_df, test_df = temporal_split(df)
    feature_cols = [c for c in df.columns if c != "Class"]

    X_train = train_df[feature_cols].values
    y_train = train_df["Class"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["Class"].values

    # ---------- modelo 1: LR + class_weight ----------
    print("\nEntrenando LR + class_weight='balanced'...")
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    pipe_lr.fit(X_train, y_train)
    sv_lr = pipe_lr.predict_proba(X_val)[:, 1]
    thr_lr = tune_threshold_for_cost(y_val, sv_lr)
    print(f"  Umbral tuneado en val: {thr_lr:.4f}")

    joblib.dump(pipe_lr, MODELS_DIR / "lr_class_weight.joblib")
    print(f"  Guardado: {MODELS_DIR / 'lr_class_weight.joblib'}")

    # ---------- modelo 2: NN + Focal Loss (gamma=2) ----------
    print("\nEntrenando NN + Focal Loss (gamma=2, alpha=0.25)...")
    scaler_nn = StandardScaler()
    X_train_sc = scaler_nn.fit_transform(X_train)
    X_val_sc = scaler_nn.transform(X_val)

    focal = FocalLossBinary(alpha=0.25, gamma=2.0)
    model_nn = train_nn(X_train_sc, y_train, focal, epochs=20)

    model_nn.eval()
    with torch.no_grad():
        sv_nn = torch.sigmoid(
            model_nn(torch.tensor(X_val_sc, dtype=torch.float32))
        ).numpy()
    thr_nn = tune_threshold_for_cost(y_val, sv_nn)
    print(f"  Umbral tuneado en val: {thr_nn:.4f}")

    torch.save({
        "state_dict": model_nn.state_dict(),
        "in_dim": X_train.shape[1],
        "scaler_mean": scaler_nn.mean_.tolist(),
        "scaler_scale": scaler_nn.scale_.tolist(),
    }, MODELS_DIR / "nn_focal_loss.pt")
    print(f"  Guardado: {MODELS_DIR / 'nn_focal_loss.pt'}")

    # ---------- metadatos ----------
    test_df.to_parquet(MODELS_DIR / "test_set.parquet", index=False)
    print(f"  Guardado test set para el demo: {MODELS_DIR / 'test_set.parquet'}")

    meta = {
        "feature_cols": feature_cols,
        "thresholds_tuned_on_val": {
            "lr_class_weight": thr_lr,
            "nn_focal_loss": thr_nn,
        },
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "test_positives": int(test_df["Class"].sum()),
    }
    with (MODELS_DIR / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Guardado: {MODELS_DIR / 'meta.json'}")
    print("\nListo. Ahora podes correr: streamlit run app.py")


if __name__ == "__main__":
    main()
