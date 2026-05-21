"""
Focal Loss para clasificacion binaria desbalanceada.

Referencia: Lin et al. (2017), "Focal Loss for Dense Object Detection".
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

donde p_t es la probabilidad estimada para la clase verdadera. El factor
(1 - p_t)^gamma "aplasta" la perdida de los ejemplos faciles (los que ya
estan bien clasificados con alta confianza) y deja casi intacta la de los
dificiles, asi el gradiente se concentra donde realmente importa.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossBinary(nn.Module):
    """Focal loss para clasificacion binaria con logits.

    Parameters
    ----------
    alpha : float | None
        Peso de balance entre clases. Si None, no aplica re-balance por clase
        (equivale a usar solo el factor de focusing).
    gamma : float
        Parametro de focusing. gamma=0 es cross-entropy normal. Valores tipicos
        entre 1 y 5; el paper usa gamma=2.
    reduction : {'mean', 'sum', 'none'}
        Como agregar la perdida sobre el batch.
    """

    def __init__(self, alpha: float | None = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE con logits para estabilidad numerica
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        # p_t = p si y=1, (1-p) si y=0
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = focal_factor * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def focal_loss_xgb_objective(alpha: float = 0.25, gamma: float = 2.0):
    """Devuelve un objetivo custom de focal loss para XGBoost.

    Uso:
        obj = focal_loss_xgb_objective(alpha=0.25, gamma=2.0)
        model = xgb.XGBClassifier(objective=obj, ...)

    Notas:
        XGBoost necesita las derivadas primera (grad) y segunda (hess).
        Aqui las derivamos en forma cerrada para focal loss binaria.
    """

    def _objective(y_true: np.ndarray, y_pred: np.ndarray):
        # y_pred son raw scores (logits)
        p = 1.0 / (1.0 + np.exp(-y_pred))
        # gradiente y hessiana de FL respecto al logit z
        # ver derivacion en notebook (seccion focal loss)
        y = y_true
        p_t = p * y + (1 - p) * (1 - y)
        a_t = alpha * y + (1 - alpha) * (1 - y)
        g = gamma

        # d/dz FL: simplificada despues de bastante algebra
        term1 = a_t * (1 - p_t) ** g
        term2 = g * p_t * np.log(np.clip(p_t, 1e-8, 1.0)) + p_t - 1
        grad = term1 * term2 * (1 if True else 1)  # signo absorbido abajo
        # signo: derivada respecto a y=1 vs y=0
        sign = 2 * y - 1  # +1 si y=1, -1 si y=0
        grad = -sign * term1 * (g * p_t * np.log(np.clip(p_t, 1e-8, 1.0)) + p_t - 1)

        # hessiana aproximada (forma de Newton estable)
        hess = term1 * (
            (1 - p_t) * (g * np.log(np.clip(p_t, 1e-8, 1.0)) + 1)
            + g * p_t * (g * np.log(np.clip(p_t, 1e-8, 1.0)) + 2 - 1 / np.clip(p_t, 1e-8, 1.0))
        )
        hess = np.abs(hess) + 1e-6  # estabilizar

        return grad, hess

    return _objective
