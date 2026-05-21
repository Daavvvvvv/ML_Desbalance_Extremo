"""Helpers de evaluacion para problemas de desbalance extremo."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    name: str
    pr_auc: float
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_row(self) -> dict:
        return {
            "modelo": self.name,
            "PR-AUC": round(self.pr_auc, 4),
            "ROC-AUC": round(self.roc_auc, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "Umbral": round(self.threshold, 3),
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
        }


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> EvalResult:
    """Calcula metricas relevantes para clasificacion binaria desbalanceada."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return EvalResult(
        name=name,
        pr_auc=average_precision_score(y_true, y_score),
        roc_auc=roc_auc_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def best_threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.5) -> float:
    """Encuentra el umbral que maximiza recall sujeto a precision >= min_precision."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve devuelve len(thr) = len(prec) - 1
    valid = prec[:-1] >= min_precision
    if not valid.any():
        return 0.5
    candidate_recall = rec[:-1][valid]
    candidate_thr = thr[valid]
    return float(candidate_thr[np.argmax(candidate_recall)])


def cost_score(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 100.0) -> float:
    """Costo total asumiendo cost_fp por falso positivo y cost_fn por falso negativo.

    En fraude: dejar pasar un fraude (FN) cuesta mucho mas que revisar
    una transaccion legitima por error (FP). Default 1:100.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fp * cost_fp + fn * cost_fn)


def plot_pr_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], ax=None):
    """Plotea curvas PR superpuestas para comparar modelos.

    Parameters
    ----------
    results : dict
        {nombre_modelo: (y_true, y_score)}
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, (y_true, y_score) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", linewidth=2)
    # linea horizontal de baseline aleatorio (proporcion de positivos)
    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5, label=f"random (AP={baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return ax


def summary_table(results: list[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in results]).set_index("modelo")
