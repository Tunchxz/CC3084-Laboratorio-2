"""
Extracción de las características de catch22.

`pycatch22` es la implementación oficial en C del conjunto catch22 (*CAnonical
Time-series CHaracteristics*). Recibe una lista de números y devuelve 22 valores
que resumen el comportamiento de la serie: autocorrelación, forma de la
distribución, periodicidad, rachas, entropía, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pycatch22


def caracteristicas(valores) -> dict[str, float]:
    """Calcula las 22 características de una sola serie y las devuelve como dict."""
    resultado = pycatch22.catch22_all(list(np.asarray(valores, dtype=float)))
    return dict(zip(resultado["names"], resultado["values"]))


def matriz_catch22(series: dict[str, pd.Series], nombres: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Construye la matriz del inciso 2.3: una fila por serie, una columna por
    característica.
    """
    filas = {}
    for clave, serie in series.items():
        etiqueta = nombres[clave] if nombres else clave
        filas[etiqueta] = caracteristicas(serie.values)
    return pd.DataFrame(filas).T


def catch22_movil(serie: pd.Series, ventana: int = 24) -> pd.DataFrame:
    """
    Calcula las 22 características sobre ventanas móviles de la serie.

    Para cada mes `t` se usan los `ventana` meses que terminan en `t` (incluido).
    Como solo se miran datos pasados, estas características pueden usarse como
    variables de entrada de un modelo sin filtrar información del futuro.

    Los primeros `ventana - 1` meses quedan como NaN porque todavía no hay
    historia suficiente.
    """
    valores = serie.values.astype(float)
    filas, indices = [], []
    for t in range(ventana - 1, len(valores)):
        filas.append(caracteristicas(valores[t - ventana + 1 : t + 1]))
        indices.append(serie.index[t])

    tabla = pd.DataFrame(filas, index=indices)
    # Alguna característica puede salir indefinida en ventanas totalmente planas
    # (por ejemplo, los meses de cierre de fronteras con puros ceros).
    return tabla.reindex(serie.index).ffill().bfill()


def quitar_constantes(matriz: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Elimina las características que valen lo mismo en todas las series.

    Con solo 7 series es posible que alguna característica no varíe; al
    estandarizar produciría una división entre cero. Devuelve la matriz filtrada
    y la lista de columnas descartadas.
    """
    constantes = [c for c in matriz.columns if matriz[c].std(ddof=0) < 1e-12]
    return matriz.drop(columns=constantes), constantes
