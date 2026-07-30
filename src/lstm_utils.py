"""
Constructores de las arquitecturas LSTM (M1-M4), grid de tuneo de
hiperparámetros, y utilidades de entrenamiento.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from datos import SEED, evaluar

ARQUITECTURAS = ("M1", "M2", "M3", "M4")


def fijar_semilla(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Constructores de modelos (CLAUDE.md 5.3)
# ---------------------------------------------------------------------------

def construir_modelo(arquitectura: str, k: int, unidades: int = 32,
                      learning_rate: float = 0.001, dropout: float = 0.2) -> keras.Model:
    """
    Construye una de las 4 arquitecturas sobre ventanas univariadas de
    forma (k, 1). M4 usa la misma arquitectura que M1: la diferencia de M4
    está en el preprocesamiento (serie diferenciada), no en la red -- ver
    `preparar_datasets_m4` más abajo.
    """
    fijar_semilla()
    entrada = keras.Input(shape=(k, 1))

    if arquitectura in ("M1", "M4"):
        x = layers.LSTM(unidades, activation="tanh")(entrada)
    elif arquitectura == "M2":
        x = layers.LSTM(unidades, activation="tanh", return_sequences=True)(entrada)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(max(unidades // 2, 4), activation="tanh")(x)
    elif arquitectura == "M3":
        x = layers.Bidirectional(layers.LSTM(unidades, activation="tanh"))(entrada)
        x = layers.Dropout(dropout)(x)
    else:
        raise ValueError(f"Arquitectura desconocida: {arquitectura}. Use una de {ARQUITECTURAS}.")

    salida = layers.Dense(1)(x)
    modelo = keras.Model(inputs=entrada, outputs=salida)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return modelo


def entrenar_modelo(modelo: keras.Model, x_train, y_train, x_val, y_val,
                     epochs: int = 200, batch_size: int = 16, patience: int = 20, verbose: int = 0):
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True,
    )
    historia = modelo.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size,
        shuffle=False,  # crítico: no barajar, se rompe el orden temporal
        callbacks=[early_stop],
        verbose=verbose,
    )
    return historia


# ---------------------------------------------------------------------------
# Grid de tuneo (CLAUDE.md 5.4)
# ---------------------------------------------------------------------------

GRID_ETAPA_A = {
    "unidades": (16, 32, 64),
    "learning_rate": (0.01, 0.001),
}
K_CANDIDATOS = (6, 12, 24)
BATCH_SIZE_FIJO = 16
DROPOUT_FIJO = 0.2


def evaluar_configuracion(arquitectura: str, datasets: dict, unidades: int,
                           learning_rate: float, batch_size: int = BATCH_SIZE_FIJO,
                           dropout: float = DROPOUT_FIJO, epochs: int = 200,
                           patience: int = 20) -> dict:
    """Entrena una configuración y devuelve sus métricas de validación."""
    k = datasets["k"]
    t0 = time.time()
    modelo = construir_modelo(arquitectura, k=k, unidades=unidades,
                               learning_rate=learning_rate, dropout=dropout)
    historia = entrenar_modelo(
        modelo, datasets["x_train"], datasets["y_train"],
        datasets["x_val"], datasets["y_val"],
        epochs=epochs, batch_size=batch_size, patience=patience,
    )
    duracion = time.time() - t0

    pred_val = modelo.predict(datasets["x_val"], verbose=0).ravel()
    metricas_escaladas = evaluar(datasets["y_val"], pred_val)

    return {
        "serie": datasets["nombre"], "arquitectura": arquitectura,
        "k": k, "unidades": unidades, "learning_rate": learning_rate,
        "batch_size": batch_size, "dropout": dropout,
        "epocas_efectivas": len(historia.history["loss"]),
        "rmse_val_escalado": metricas_escaladas["RMSE"],
        "mae_val_escalado": metricas_escaladas["MAE"],
        "tiempo_seg": round(duracion, 1),
        "_modelo": modelo, "_historia": historia,
    }


def grid_search_etapa_a(arquitectura: str, construir_datasets_fn, nombre_serie: str,
                         k_candidatos=K_CANDIDATOS, grid=GRID_ETAPA_A) -> list[dict]:
    """
    Etapa A del tuneo: barre k x unidades x learning_rate para una
    arquitectura y una serie, con batch_size y dropout fijos.
    `construir_datasets_fn(nombre_serie, k)` debe devolver el dict de
    `datos.construir_datasets`.
    """
    resultados = []
    combinaciones = list(itertools.product(k_candidatos, grid["unidades"], grid["learning_rate"]))
    for k, unidades, lr in combinaciones:
        datasets = construir_datasets_fn(nombre_serie, k)
        fila = evaluar_configuracion(arquitectura, datasets, unidades, lr)
        resultados.append(fila)
    return resultados


def grid_search_etapa_b(arquitectura: str, construir_datasets_fn, nombre_serie: str,
                         mejores_etapa_a: list[dict],
                         batch_sizes=(8, 16), dropouts=(0.0, 0.2)) -> list[dict]:
    """
    Etapa B: para las mejores configuraciones (k, unidades, lr) de la etapa
    A, refina batch_size y dropout.
    """
    resultados = []
    for base in mejores_etapa_a:
        datasets = construir_datasets_fn(nombre_serie, base["k"])
        for batch_size, dropout in itertools.product(batch_sizes, dropouts):
            if batch_size == BATCH_SIZE_FIJO and dropout == DROPOUT_FIJO:
                continue  # ya se evaluó en la etapa A
            fila = evaluar_configuracion(
                arquitectura, datasets, base["unidades"], base["learning_rate"],
                batch_size=batch_size, dropout=dropout,
            )
            resultados.append(fila)
    return resultados
