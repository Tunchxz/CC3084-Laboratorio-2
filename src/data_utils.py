"""
Carga de las series de tiempo construidas en el Laboratorio 1.

El archivo `data/raw/series_mensuales.csv` es la salida del notebook
`02_preparacion_series.ipynb` del laboratorio anterior. Está en formato largo,
con una fila por (serie, mes), y trae la etiqueta `conjunto` que marca la
partición cronológica 70/30 original. Este módulo la respeta tal cual: el
Laboratorio 2 debe usar exactamente los mismos conjuntos de entrenamiento y
prueba.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_PANEL = RAIZ / "data" / "raw" / "series_mensuales.csv"

# Nombre legible de cada una de las 7 series, con la categoría a la que pertenece.
NOMBRES = {
    "total": "Total (obligatoria)",
    "region_america_central": "Región: América del Centro",
    "region_america_norte": "Región: América del Norte",
    "region_europa": "Región: Europa",
    "frontera_aurora": "Frontera: 01 La Aurora",
    "frontera_valle_nuevo": "Frontera: 07 Valle Nuevo",
    "frontera_san_cristobal": "Frontera: 09 San Cristóbal",
}

CATEGORIAS = {
    "total": "General",
    "region_america_central": "Regiones",
    "region_america_norte": "Regiones",
    "region_europa": "Regiones",
    "frontera_aurora": "Fronteras",
    "frontera_valle_nuevo": "Fronteras",
    "frontera_san_cristobal": "Fronteras",
}


def cargar_panel(ruta: Path | str = RUTA_PANEL) -> pd.DataFrame:
    """Lee el CSV en formato largo con las 7 series."""
    return pd.read_csv(ruta, parse_dates=["fecha"])


def cargar_serie(clave: str, panel: pd.DataFrame | None = None):
    """
    Devuelve `(serie, train, test)` para una de las 7 claves de `NOMBRES`.

    `serie` es la serie mensual completa con frecuencia MS; `train` y `test`
    son los mismos cortes que usó el Laboratorio 1.
    """
    if panel is None:
        panel = cargar_panel()

    sub = panel[panel["serie"] == clave].sort_values("fecha").set_index("fecha")
    serie = sub["viajeros"].asfreq("MS")
    train = serie[sub["conjunto"] == "train"]
    test = serie[sub["conjunto"] == "test"]
    return serie, train, test


def cargar_todas(panel: pd.DataFrame | None = None) -> dict[str, pd.Series]:
    """Devuelve un diccionario {clave: serie completa} con las 7 series."""
    if panel is None:
        panel = cargar_panel()
    return {clave: cargar_serie(clave, panel)[0] for clave in NOMBRES}
