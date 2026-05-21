# Datos

Esta carpeta no contiene los datasets (estan en `.gitignore`).

## Como obtener el dataset

**Credit Card Fraud Detection** — Kaggle, dataset `mlg-ulb/creditcardfraud`.

### Opcion 1: Web
1. Ir a https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Descargar `creditcard.csv` (~150 MB)
3. Colocarlo aqui: `data/raw/creditcard.csv`

### Opcion 2: Kaggle CLI
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

### Opcion 3: Fallback automatico
Si el archivo no esta presente, el notebook intentara descargarlo via `sklearn.datasets.fetch_openml('creditcardfraud')` o, en su defecto, generara un dataset sintetico equivalente para que el notebook corra de todos modos.
