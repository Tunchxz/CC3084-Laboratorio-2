# CC3084 — Laboratorio 2: Deep Learning (LSTM) y catch22

Continuación del Laboratorio 1 de series de tiempo (ingreso de viajeros
internacionales a Guatemala). Modelos LSTM con tuneo de hiperparámetros y
análisis de similitud de series con catch22.


## Reproducir el entorno

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name cc3084-lab2 --display-name "CC3084 Lab 2"
```

## Estructura

```
data/processed/        series del Laboratorio 1 (solo lectura) + data/processed/lab2 (derivados de este lab)
src/                    lógica reutilizable: carga de datos, modelos LSTM, catch22
notebooks/              notebooks numerados por ejercicio
models/                 pesos entrenados (.keras, no versionados)
results/                tablas de tuneo y comparación (versionadas, evidencia de rúbrica)
figuras/                gráficos exportados para el informe
informe/                informe final
```

## Datos

- `data/raw/` no se modifica bajo ninguna circunstancia.
- `data/processed/series_mensuales.csv`, `resumen_regiones.csv` y
  `resumen_fronteras.csv` vienen del Laboratorio 1 y son de solo lectura aquí.
- La partición train/test es la misma del Laboratorio 1 (columna `conjunto`).


