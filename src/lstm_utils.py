"""
Utilidades para los modelos LSTM (PyTorch).

El flujo es:

    serie -> logaritmo -> diferenciación -> estandarización -> ventanas -> LSTM

con tres decisiones diferentes, todas justificadas por las
características de las series con las que se trabaja en el Laboratorio 2:

1. **Logaritmo antes de diferenciar.** En el Laboratorio 1 estas series
   resultaron tener descomposición multiplicativa (la amplitud estacional crece
   con el nivel), y allí se modelaron en escala logarítmica. Se conserva ese
   criterio: el logaritmo estabiliza la varianza y evita que el desplome de 2020
   domine todo el entrenamiento.
2. **El `StandardScaler` se ajusta solo con el tramo de entrenamiento.** Si se
   ajustara con la serie completa, la media y la desviación usadas para
   normalizar ya incluirían información del futuro.
3. **La validación es un tramo interior anterior a la pandemia.** Ver
   `dividir()`.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Preparación de los datos
# ---------------------------------------------------------------------------

class PreparadorSerie:
    """
    Convierte una serie mensual en los tensores que consume la LSTM y sabe
    deshacer las transformaciones para volver a la escala de viajeros.

    Notación interna: `base` es la serie transformada (en logaritmo), `dif` son
    sus cambios mes a mes y `z` esas diferencias ya estandarizadas. La posición
    `p` de un valor en `z` corresponde al mes `p + 1` de la serie original, así
    que `base[p]` es siempre el mes anterior al que se quiere predecir.
    """

    def __init__(self, serie: pd.Series, n_train: int, seq_length: int = 12,
                 log: bool = True):
        if log and (serie <= 0).any():
            raise ValueError("La serie tiene ceros o negativos: no se puede usar logaritmo.")

        self.serie = serie
        self.n_train = n_train
        self.seq_length = seq_length
        self.log = log

        self.base = np.log(serie) if log else serie.copy()
        self.dif = self.base.diff().dropna()
        # Las primeras `n_train - 1` diferencias caen dentro del entrenamiento.
        self.n_dif_train = n_train - 1

        self.scaler = StandardScaler().fit(
            self.dif.iloc[: self.n_dif_train].values.reshape(-1, 1)
        )
        self.z = self.scaler.transform(self.dif.values.reshape(-1, 1)).ravel()

    # -- ventanas ----------------------------------------------------------

    def ventanas(self):
        """
        Construye todas las ventanas supervisadas de la serie.

        Devuelve `(X, y, posiciones)`, donde `X` tiene forma
        (muestras, pasos de tiempo, características), `y` es el valor siguiente
        y `posiciones` indica en qué punto de `z` cae cada objetivo.
        """
        x, y, pos = [], [], []
        for i in range(len(self.z) - self.seq_length):
            x.append(self.z[i : i + self.seq_length])
            y.append(self.z[i + self.seq_length])
            pos.append(i + self.seq_length)
        x = np.array(x, dtype=np.float32)[:, :, None]
        return x, np.array(y, dtype=np.float32), np.array(pos)

    def fechas_objetivo(self, posiciones: np.ndarray) -> pd.DatetimeIndex:
        """Mes que se predice en cada posición."""
        return self.serie.index[np.asarray(posiciones) + 1]

    def dividir(self, inicio_val: str = "2018-01-01", fin_val: str = "2019-12-01") -> dict:
        """
        Separa las ventanas en ajuste, validación y prueba.

        La **prueba** es exactamente la del Laboratorio 1 y no se toca.

        La **validación** es un tramo interior del entrenamiento, anterior a la
        pandemia. La alternativa natural —usar el final del entrenamiento— no
        funciona aquí: ese tramo es justo el desplome de 2020, que ningún modelo
        entrenado con datos previos puede anticipar, así que validar ahí no
        distingue un buen modelo de uno malo.

        El **ajuste** son las ventanas anteriores a la validación. Los meses de
        entrenamiento posteriores a la validación (la pandemia) no se usan en
        esta etapa, pero sí en el reajuste final, para el que se devuelve
        también el conjunto de entrenamiento completo.
        """
        x, y, pos = self.ventanas()
        fechas = self.fechas_objetivo(pos)
        en_train = pos < self.n_dif_train
        es_val = en_train & (fechas >= inicio_val) & (fechas <= fin_val)
        es_ajuste = en_train & (fechas < inicio_val)

        return {
            "x_ajuste": x[es_ajuste], "y_ajuste": y[es_ajuste], "pos_ajuste": pos[es_ajuste],
            "x_val": x[es_val], "y_val": y[es_val], "pos_val": pos[es_val],
            "x_train": x[en_train], "y_train": y[en_train], "pos_train": pos[en_train],
            "x_test": x[~en_train], "y_test": y[~en_train], "pos_test": pos[~en_train],
        }

    def ventana_en(self, pos_objetivo: int) -> np.ndarray:
        """Ventana de entrada que termina justo antes de la posición dada."""
        return self.z[pos_objetivo - self.seq_length : pos_objetivo].copy()

    def ventana_final_train(self) -> np.ndarray:
        """Última ventana del entrenamiento: punto de partida del pronóstico
        recursivo sobre el conjunto de prueba."""
        return self.ventana_en(self.n_dif_train)

    # -- reversión de las transformaciones ---------------------------------

    def _a_viajeros(self, valores_base):
        return np.exp(valores_base) if self.log else valores_base

    def revertir_un_paso(self, z_pred, posiciones) -> np.ndarray:
        """
        Pasa predicciones de un paso adelante a la escala de viajeros.

        Cada predicción es una diferencia estandarizada: se desestandariza y se
        le suma el valor **real** del mes anterior.
        """
        dif_pred = self.scaler.inverse_transform(np.asarray(z_pred).reshape(-1, 1)).ravel()
        return self._a_viajeros(self.base.values[np.asarray(posiciones)] + dif_pred)

    def revertir_recursivo(self, z_pred, pos_ancla: int | None = None) -> np.ndarray:
        """
        Pasa un pronóstico recursivo a la escala de viajeros.

        Aquí no hay valores reales que usar mes a mes: se parte del último valor
        conocido (`pos_ancla`, por defecto el final del entrenamiento) y se
        acumulan las diferencias predichas.
        """
        if pos_ancla is None:
            pos_ancla = self.n_dif_train
        dif_pred = self.scaler.inverse_transform(np.asarray(z_pred).reshape(-1, 1)).ravel()
        return self._a_viajeros(self.base.values[pos_ancla] + np.cumsum(dif_pred))


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------

class ModeloLSTM(nn.Module):
    """
    Red LSTM para pronóstico univariado o multivariado.

    `hidden_size` es cuántas neuronas tiene la capa recurrente, `num_layers`
    cuántas capas LSTM se apilan y `dropout` cuánta regularización se aplica
    entre capas. La capa `Linear` final tiene una sola salida porque se predice
    un único valor: el mes siguiente.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 20,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        salida, _ = self.lstm(x)          # (lote, pasos, hidden)
        return self.fc(salida[:, -1, :])  # se usa solo el último paso


def contar_parametros(modelo) -> int:
    """Número de pesos que la red tiene que aprender."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def entrenar(modelo, x_train, y_train, x_val, y_val, epocas: int = 400,
             lr: float = 0.01, batch_size: int = 16, paciencia: int = 40,
             semilla: int = 42, verbose: bool = False) -> dict:
    """
    Entrena el modelo y devuelve el historial de pérdidas.

    Usa error cuadrático medio como pérdida, el optimizador Adam y detención
    temprana: si la pérdida de validación no mejora durante `paciencia` épocas,
    el entrenamiento se corta y se restauran los pesos de la mejor época.

    Para el reajuste final se puede pasar `paciencia=None`, que desactiva la
    detención temprana y entrena el número exacto de épocas indicado.
    """
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    x_tr = torch.from_numpy(x_train).float()
    y_tr = torch.from_numpy(y_train).float().unsqueeze(1)
    x_va = torch.from_numpy(x_val).float()
    y_va = torch.from_numpy(y_val).float().unsqueeze(1)

    cargador = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr, y_tr),
        batch_size=batch_size, shuffle=False,
    )
    funcion_perdida = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    perdidas_train, perdidas_val = [], []
    mejor_val, mejor_estado, sin_mejora, mejor_epoca = np.inf, None, 0, 0

    for epoca in range(epocas):
        modelo.train()
        acumulado = 0.0
        for xb, yb in cargador:
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
            acumulado += perdida.item() * xb.size(0)
        perdidas_train.append(acumulado / len(cargador.dataset))

        modelo.eval()
        with torch.no_grad():
            perdida_val = funcion_perdida(modelo(x_va), y_va).item()
        perdidas_val.append(perdida_val)

        if perdida_val < mejor_val - 1e-7:
            mejor_val, mejor_epoca = perdida_val, epoca
            if paciencia is not None:
                mejor_estado = copy.deepcopy(modelo.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if paciencia is not None and sin_mejora >= paciencia:
                break

        if verbose and (epoca + 1) % 25 == 0:
            print(f"  época {epoca + 1:3d} | train {perdidas_train[-1]:.4f} | val {perdida_val:.4f}")

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    return {
        "perdidas_train": perdidas_train,
        "perdidas_val": perdidas_val,
        "mejor_val": mejor_val,
        "mejor_epoca": mejor_epoca,
        "epocas_corridas": len(perdidas_train),
    }


def graficar_perdidas(historial: dict, titulo: str):
    """Curvas de pérdida de entrenamiento y validación."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(historial["perdidas_train"], label="Ajuste")
    ax.plot(historial["perdidas_val"], label="Validación")
    ax.axvline(historial["mejor_epoca"], color="grey", ls="--", lw=1,
               label=f"Mejor época ({historial['mejor_epoca']})")
    ax.set(title=titulo, xlabel="Época", ylabel="Pérdida (MSE)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predecir(modelo, x: np.ndarray) -> np.ndarray:
    """Predicción de un paso adelante sobre ventanas ya construidas."""
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.from_numpy(x).float()).numpy().ravel()


def predecir_recursivo(modelo, ventana_inicial: np.ndarray, pasos: int,
                       extras: np.ndarray | None = None) -> np.ndarray:
    """
    Pronóstico a varios meses realimentando el modelo con sus propias salidas.

    Es lo que hacen ARIMA, Holt-Winters y Prophet en el Laboratorio 1:
    pronostican todo el horizonte sin volver a ver datos reales. Por eso es la
    forma justa de comparar la LSTM contra ellos.

    `extras`, si se pasa, son variables adicionales conocidas por paso, de forma
    (pasos, ventana, n_extras); se usa en el notebook de catch22.
    """
    modelo.eval()
    ventana = list(np.asarray(ventana_inicial, dtype=np.float32))
    predicciones = []
    with torch.no_grad():
        for paso in range(pasos):
            columna = np.array(ventana, dtype=np.float32).reshape(1, -1, 1)
            if extras is not None:
                columna = np.concatenate([columna, extras[paso][None, :, :]], axis=2)
            siguiente = modelo(torch.from_numpy(columna).float()).item()
            predicciones.append(siguiente)
            ventana.pop(0)
            ventana.append(siguiente)
    return np.array(predicciones)
