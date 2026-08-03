# Laboratorio 2: Deep Learning

Modelos **LSTM** para pronosticar el ingreso mensual de viajeros internacionales a Guatemala,
y exploración de la similitud entre las series con el algoritmo **catch22**.

Este laboratorio continúa el Laboratorio 1, reutiliza sus series ya construidas y sus mismos
conjuntos de entrenamiento y prueba.

## Estructura

```bash
Laboratorio-2/
├── notebooks/
│   ├── 01_lstm_series.ipynb      # Ejercicio 1 — modelos LSTM y comparación contra el Lab 1
│   ├── 02_catch22.ipynb          # Ejercicio 2, incisos 2.1–2.13 — similitud entre series
│   └── 03_lstm_catch22.ipynb     # Ejercicio 2, inciso 2.14 — LSTM alimentada con catch22
├── src/
│   ├── data_utils.py             # Carga de las 7 series y su partición
│   ├── lstm_utils.py             # Preparación, modelo LSTM, entrenamiento y predicción
│   ├── catch22_utils.py          # Extracción de características, global y en ventana móvil
│   └── ts_utils.py               # Copiado del Laboratorio 1 (métricas y caracterización)
├── data/
│   ├── raw/                      # Salidas del Laboratorio 1
│   └── processed/                # Generado por los notebooks
├── requirements.txt
└── README.md
```

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Orden de ejecución

Los notebooks son secuenciales y se ejecutan desde la carpeta `notebooks/`:

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_lstm_series.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 02_catch22.ipynb
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 03_lstm_catch22.ipynb
```

| # | Notebook | Qué hace | Requiere | Produce en `data/processed/` |
|---|---|---|---|---|
| 1 | `01_lstm_series.ipynb` | Dos configuraciones de LSTM por serie, tuneo, predicción y comparación contra el Laboratorio 1 | `series_mensuales.csv`, `resumen_total.csv`, `resumen_regiones.csv` | `comparacion_lab1_lab2.csv`, `mejor_config_lstm.json` |
| 2 | `02_catch22.ipynb` | Extracción de las 22 características de las 7 series, PCA, clustering, heatmaps, distancias e interpretación | `series_mensuales.csv` | `matriz_catch22.csv`, `grupos_catch22.csv`, `importancia_catch22.csv` |
| 3 | `03_lstm_catch22.ipynb` | LSTM multivariada con catch22 en ventana móvil, comparada contra la LSTM univariada | Las salidas de los notebooks 1 y 2 | `comparacion_lstm_catch22.csv` |

El notebook 3 depende de los dos anteriores, toma de ellos la configuración ganadora y el
orden de importancia de las características.

## Procedencia de los datos en `data/raw/`

Todos son salidas del Laboratorio 1 y aquí se tratan como datos crudos. No se modifican.

| Archivo | Origen | Contenido |
|---|---|---|
| `series_mensuales.csv` | `Laboratorio-1/data/processed/` | Las 7 series mensuales en formato largo (`serie`, `fecha`, `conjunto`, `viajeros`), 2009-01 a 2026-06, con la partición 70/30 original |
| `resumen_regiones.csv` | `Laboratorio-1/data/processed/` | Mejor modelo y métricas del Laboratorio 1 para las 3 regiones |
| `resumen_fronteras.csv` | `Laboratorio-1/data/processed/` | Ídem para las 3 fronteras |
| `resumen_total.csv` | **Creado para este laboratorio** | Métricas de los 5 modelos clásicos de la serie total. El notebook `03_serie_total.ipynb` del Laboratorio 1 nunca exportó su tabla a CSV, así que se transcribieron los valores de la salida ya ejecutada de ese notebook |

Los valores de `resumen_total.csv` son los que ese notebook mostró redondeados a viajeros
enteros; es precisión más que suficiente para comparar modelos cuyos errores están en el
orden de las decenas de miles.
