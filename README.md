# Exposicion: Desbalance extremo — Focal Loss y Cost-Sensitive Pipelines

Topico avanzado del curso **Aprendizaje de Maquina Aplicado** (ST1613/ST1631, EAFIT, 2026-1).
Profesor: Marco Teran (`mtteranl@eafit.edu.co`).

## Equipo

- David Franco
- (compañero 2)
- (compañero 3)

## De que va

Cuando una clase es muy rara (fraudes, enfermedades, fallas), los modelos tienden a ignorarla y los `accuracy` se vuelven engañosos. Esta expo cubre dos familias de tecnicas para atacar ese problema:

1. **Cost-sensitive learning** — meterle peso a la clase minoritaria via `class_weight`, `sample_weight` o `scale_pos_weight`, mas ajuste de umbral.
2. **Focal Loss** — modificar la funcion de perdida para que el modelo se enfoque en los ejemplos dificiles y deje de gastar capacidad en los faciles. Viene del paper de RetinaNet (Lin et al., 2017).

## Dataset

**Credit Card Fraud Detection** (Kaggle / `mlg-ulb`).

- 284,807 transacciones, 492 fraudes (0.172%).
- 28 features anonimizadas via PCA + `Time` + `Amount`.
- Es el ejemplo canonico de desbalance extremo en ML.

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Estructura

```
Exposicion ML/
├── README.md                    este archivo
├── requirements.txt
├── .gitignore
├── app.py                       demo interactivo en Streamlit
├── data/
│   └── raw/                     creditcard.csv (NO versionado)
├── notebooks/
│   └── desbalance_extremo.ipynb notebook principal (11 secciones, expo + analisis)
├── src/
│   ├── focal_loss.py            implementacion limpia de focal loss
│   ├── metrics.py               helpers de evaluacion
│   └── train_models.py          entrena y guarda los 2 mejores modelos
├── models/                      modelos entrenados (NO versionados)
├── figures/                     graficos exportados para slides
└── .claude/agents/              agentes especializados
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

## Descargar el dataset

**Opcion web:**
1. Ir a https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descargar `creditcard.csv` (~150 MB)
3. Colocarlo en `data/raw/creditcard.csv`

**Opcion CLI (requiere Kaggle API):**
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

## Ejecutar

**1) El notebook completo (analisis y resultados):**
```bash
jupyter lab notebooks/desbalance_extremo.ipynb
```

**2) Entrenar los modelos para el demo (una sola vez):**
```bash
python -m src.train_models
```

**3) Demo interactivo en Streamlit:**
```bash
streamlit run app.py
```

El demo permite:
- Cambiar el umbral de decision en vivo y ver como cambian recall/precision/costo
- Modificar los costos asimetricos (FP vs FN) y reoptimizar
- Probar muestras individuales (transacciones reales del test set)
- Comparar LR + class_weight vs NN + Focal Loss

## Estructura de la exposicion (10 min)

| Bloque | Tiempo | Persona | Contenido |
|---|---|---|---|
| Contexto y problema | 3 min | 1 | Desbalance extremo, accuracy paradox, metricas correctas |
| Cost-sensitive | 3.5 min | 2 | `class_weight`, `scale_pos_weight`, umbral, demo |
| Focal Loss | 3.5 min | 3 | Matematica, intuicion, implementacion, comparacion |

## Referencias

- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). ICCV.
- He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE TKDE.
- scikit-learn docs: [class_weight](https://scikit-learn.org/stable/modules/svm.html#unbalanced-problems)
- XGBoost docs: [scale_pos_weight](https://xgboost.readthedocs.io/en/stable/parameter.html)
