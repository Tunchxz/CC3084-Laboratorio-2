"""
Carga de series, partición train/test fija (heredada del Laboratorio 1) y
transformaciones (log1p, escalado, ventanas deslizantes) con sus inversas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SEED = 40

# Rutas relativas a la raíz del repositorio (los notebooks corren desde notebooks/).
RAIZ = Path(__file__).resolve().parent.parent
RUTA_SERIES = RAIZ / "data" / "processed" / "series_mensuales.csv"
RUTA_RESUMEN_REGIONES = RAIZ / "data" / "processed" / "resumen_regiones.csv"
RUTA_RESUMEN_FRONTERAS = RAIZ / "data" / "processed" / "resumen_fronteras.csv"

VENTANA_PANDEMIA = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"))

# La fila de la serie "total" resultado tomado del Lab 1,
BASELINE_TOTAL_LAB1 = {
    "serie": "total",
    "mejor_modelo": "Prophet",
    "MAE": 132304.0,
    "RMSE": 139549.0,
}


# ---------------------------------------------------------------------------
# Carga de series y partición train/test (fija, heredada del Lab 1)
# ---------------------------------------------------------------------------

def cargar_panel(ruta: Path = RUTA_SERIES) -> pd.DataFrame:
    """Carga el CSV largo/tidy del Lab 1 con las 7 series mensuales."""
    panel = pd.read_csv(ruta, parse_dates=["fecha"])
    return panel


def cargar_serie(nombre: str, panel: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Devuelve (train, test) para una serie, como pd.Series con índice de
    fecha y frecuencia mensual. La partición viene de la columna `conjunto`
    del CSV del Lab 1: nunca se recalcula ni se baraja.
    """
    if panel is None:
        panel = cargar_panel()
    sub = panel.loc[panel["serie"] == nombre].sort_values("fecha")
    if sub.empty:
        disponibles = sorted(panel["serie"].unique())
        raise ValueError(f"Serie '{nombre}' no encontrada. Disponibles: {disponibles}")

    sub = sub.set_index("fecha")
    serie = sub["viajeros"].asfreq("MS")  # las fechas ya vienen en día 1 (MS) desde el Lab 1

    train = serie.loc[sub.index[sub["conjunto"] == "train"]]
    test = serie.loc[sub.index[sub["conjunto"] == "test"]]
    return train, test


def cargar_baseline_lab1() -> pd.DataFrame:
    """
    Consolida las métricas del Lab 1 para las series de este laboratorio:
    toma las filas de `resumen_regiones.csv` y `resumen_fronteras.csv`, y
    agrega la fila de `total` (dato tomado del lab 1). 
    """
    regiones = pd.read_csv(RUTA_RESUMEN_REGIONES)
    fronteras = pd.read_csv(RUTA_RESUMEN_FRONTERAS)

    filas = []
    for _, r in regiones.iterrows():
        clave = "region_" + (
            r["región"].lower()
            .replace("américa del centro", "america_central")
            .replace("américa del norte", "america_norte")
            .replace(" ", "_")
        )
        filas.append({"serie": clave, "mejor_modelo": r["mejor_modelo"], "MAE": r["MAE"], "RMSE": r["RMSE"]})
    for _, r in fronteras.iterrows():
        mapa = {"01 La Aurora": "frontera_aurora", "07 Valle Nuevo": "frontera_valle_nuevo",
                "09 San Cristóbal": "frontera_san_cristobal"}
        clave = mapa.get(r["frontera"], "frontera_" + r["frontera"])
        filas.append({"serie": clave, "mejor_modelo": r["mejor_modelo"], "MAE": r["MAE"], "RMSE": r["RMSE"]})
    filas.append(dict(BASELINE_TOTAL_LAB1))

    return pd.DataFrame(filas).set_index("serie")


# ---------------------------------------------------------------------------
# Transformaciones (con inversas) 
# ---------------------------------------------------------------------------

def ajustar_transformacion(train: pd.Series) -> MinMaxScaler:
    """Ajusta el MinMaxScaler sobre log1p(train). Nunca ver test aquí."""
    log_train = np.log1p(train.values).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(log_train)
    return scaler


def aplicar_transformacion(valores: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """log1p -> escalado. `valores` es un array 1D en unidades originales."""
    log_vals = np.log1p(np.asarray(valores, dtype=float)).reshape(-1, 1)
    return scaler.transform(log_vals).ravel()


def invertir_transformacion(valores_escalados: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inversa exacta de `aplicar_transformacion`: escalado inverso -> expm1."""
    log_vals = scaler.inverse_transform(np.asarray(valores_escalados, dtype=float).reshape(-1, 1)).ravel()
    return np.expm1(log_vals)


def verificar_roundtrip(train: pd.Series, scaler: MinMaxScaler, tol: float = 1e-6) -> None:
    """
    Prueba de round-trip obligatoria: transformar e invertir
    debe devolver los valores originales. Lanza AssertionError si no.
    """
    escalado = aplicar_transformacion(train.values, scaler)
    recuperado = invertir_transformacion(escalado, scaler)
    error_max = np.max(np.abs(recuperado - train.values))
    assert error_max < tol, f"Round-trip de la transformación falló: error máximo = {error_max}"


# ---------------------------------------------------------------------------
# Ventanas deslizantes
# ---------------------------------------------------------------------------

def crear_ventanas(valores_escalados: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte una serie escalada 1D en pares (X, y) para aprendizaje
    supervisado: los últimos k valores predicen el siguiente.
    Devuelve X con forma (n-k, k, 1), y con forma (n-k,).
    """
    X, y = [], []
    for i in range(len(valores_escalados) - k):
        X.append(valores_escalados[i:i + k])
        y.append(valores_escalados[i + k])
    X = np.array(X).reshape(-1, k, 1)
    y = np.array(y)
    return X, y


def construir_datasets(nombre_serie: str, k: int, val_meses: int = 24,
                        panel: pd.DataFrame | None = None) -> dict:
    """
    Pipeline completo para una serie: carga train/test, ajusta la
    transformación solo con train, construye ventanas, y separa las
    últimas `val_meses` ventanas de train como validación (hold-out
    cronológico, no aleatorio).

    Devuelve un dict con todo lo necesario para entrenar, tunear, predecir
    e invertir: x_train, y_train, x_val, y_val, scaler, train, test,
    valores_train_escalados (serie completa de train ya transformada, para
    poder armar la última ventana y pronosticar de forma recursiva).
    """
    train, test = cargar_serie(nombre_serie, panel=panel)
    scaler = ajustar_transformacion(train)
    verificar_roundtrip(train, scaler)

    train_escalado = aplicar_transformacion(train.values, scaler)
    X_all, y_all = crear_ventanas(train_escalado, k)

    n_val = val_meses
    if len(y_all) <= n_val:
        raise ValueError(
            f"k={k} deja muy pocas ventanas de entrenamiento tras separar "
            f"{n_val} meses de validación (solo hay {len(y_all)} ventanas en total)."
        )

    x_train, y_train = X_all[:-n_val], y_all[:-n_val]
    x_val, y_val = X_all[-n_val:], y_all[-n_val:]

    return {
        "nombre": nombre_serie, "k": k,
        "train": train, "test": test,
        "scaler": scaler, "train_escalado": train_escalado,
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
    }


# ---------------------------------------------------------------------------
# Pronóstico
# ---------------------------------------------------------------------------

def pronostico_recursivo(modelo, ultima_ventana_escalada: np.ndarray, pasos: int) -> np.ndarray:
    """
    Pronóstico multi-paso realimentando las propias predicciones del
    modelo, sin usar ningún valor real del conjunto de prueba. Es el único
    protocolo comparable contra los modelos del Lab 1.

    `ultima_ventana_escalada`: array 1D de longitud k (últimos k valores
    escalados del train completo).
    """
    k = len(ultima_ventana_escalada)
    ventana = ultima_ventana_escalada.copy().astype(float)
    predicciones = np.zeros(pasos)
    for t in range(pasos):
        entrada = ventana.reshape(1, k, 1)
        yhat = modelo.predict(entrada, verbose=0)[0, 0]
        predicciones[t] = yhat
        ventana = np.append(ventana[1:], yhat)
    return predicciones


def pronostico_un_paso(modelo, serie_completa_escalada: np.ndarray, k: int, n_train: int) -> np.ndarray:
    """
    Diagnóstico secundario (walk-forward): predice cada mes de prueba usando
    los k valores REALES anteriores (sin realimentar predicciones). No es
    comparable contra el Lab 1 -- debe reportarse por separado.
    """
    n_total = len(serie_completa_escalada)
    n_test = n_total - n_train
    predicciones = np.zeros(n_test)
    for i, t in enumerate(range(n_train, n_total)):
        ventana = serie_completa_escalada[t - k:t].reshape(1, k, 1)
        predicciones[i] = modelo.predict(ventana, verbose=0)[0, 0]
    return predicciones


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def evaluar(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_real, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
    resultado = {"MAE": mae, "RMSE": rmse}
    if np.all(y_real != 0):
        resultado["MAPE"] = float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)
    else:
        resultado["MAPE"] = np.nan
    return resultado
