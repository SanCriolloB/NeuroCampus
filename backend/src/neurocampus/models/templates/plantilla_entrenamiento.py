"""
neurocampus.models.templates.plantilla_entrenamiento
====================================================

Template Method (Plantilla) para orquestar el entrenamiento de modelos
y emitir eventos compatibles con la UI (pestaña **Modelos**) en NeuroCampus.

Esta plantilla es responsable de:
- Ejecutar ``setup()`` de la estrategia (carga/preparación de datos).
- Ejecutar el bucle de épocas y llamar a ``train_step(...)``.
- Emitir eventos:
  - ``training.started``
  - ``training.epoch_end``
  - ``training.completed``
  - ``training.failed``
- Devolver un payload con ``history`` y ``metrics`` para que el backend
  lo exponga vía ``GET /modelos/estado/{job_id}``.

Cambios
----------------
Se amplía el contrato de la estrategia para permitir:

- ``train_step(epoch, hparams, y=None) -> metrics``
  donde el ``loss`` se devuelve dentro del diccionario de métricas
  (p.ej. ``metrics["loss"]``).

Además, se mantiene compatibilidad con implementaciones legacy que:
- aceptan solo ``train_step(epoch)``, y/o
- retornan ``(loss, metrics)``.

En todos los casos, esta plantilla:
- Normaliza ``hparams`` (keys a minúsculas).
- Extrae ``loss`` de forma robusta.
- Publica eventos con métricas enriquecidas (incluye ``time_epoch_ms``).

.. note::
   La UI típicamente grafica valores numéricos por época (loss/accuracy/etc).
   Métricas no numéricas (p.ej. ``confusion_matrix``) se mantienen en ``metrics``
   finales y se envían por evento, pero no se fuerzan a ``history``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from ..observer.eventos_entrenamiento import (
    emit_training_started,
    emit_epoch_end,
    emit_training_completed,
    emit_training_failed,
)


# -------------------------------------------------------------------
# Contrato de estrategia (compatible con implementaciones actuales y futuras)
# -------------------------------------------------------------------

TrainStepReturn = Union[
    Dict[str, Any],                   # nuevo: solo metrics (incluye loss adentro)
    Tuple[float, Dict[str, Any]],     # legacy: (loss, metrics)
]


@runtime_checkable
class EstrategiaEntrenamiento(Protocol):
    """
    Contrato mínimo esperado para estrategias de entrenamiento.

    Las estrategias deben implementar:
    - :meth:`setup` para preparar el modelo y cargar datos.
    - :meth:`train_step` para ejecutar una época y devolver métricas.

    Firma recomendada (nueva)
    -------------------------
    .. code-block:: python

        def train_step(self, epoch: int, hparams: Dict[str, Any], y: Any = None) -> Dict[str, Any]:
            return {"loss": 0.123, "accuracy": 0.9, "val_f1_macro": 0.8}

    Compatibilidad (legacy)
    -----------------------
    Se soporta también:

    .. code-block:: python

        def train_step(self, epoch: int) -> Tuple[float, Dict[str, Any]]:
            return loss, {"accuracy": 0.9}
    """

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        """Prepara modelo/datos (una sola vez antes del bucle de épocas)."""
        ...

    def train_step(self, epoch: int, *args: Any, **kwargs: Any) -> TrainStepReturn:
        """Ejecuta una época. Puede retornar metrics o (loss, metrics)."""
        ...


# -------------------------------------------------------------------
# Plantilla de entrenamiento
# -------------------------------------------------------------------

class PlantillaEntrenamiento:
    """
    Orquestador de entrenamiento (Template Method).

    Esta clase no implementa el algoritmo del modelo: delega en una estrategia,
    pero estandariza:

    - Normalización de hiperparámetros.
    - Extracción de métricas y construcción de ``history``.
    - Emisión de eventos para la UI.
    - Manejo de errores y estado final.

    :param estrategia: Implementación concreta (RBM/DBM/etc).
    """

    def __init__(self, estrategia: EstrategiaEntrenamiento):
        self.estrategia = estrategia

    # ----------------------------------------------------------
    # Normalización homogénea de hiperparámetros
    # ----------------------------------------------------------
    def _normalize_hparams(self, hparams: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normaliza hiperparámetros a un formato consistente.

        - Convierte keys a string en minúsculas.
        - Si ``hparams`` es None, retorna dict vacío.

        :param hparams: Hparams del request.
        :return: Hparams normalizados.
        """
        return {str(k).lower(): v for k, v in (hparams or {}).items()}

    def _call_train_step(self, epoch: int, hparams: Dict[str, Any], y: Any = None) -> TrainStepReturn:
        """
        Llama a ``estrategia.train_step`` soportando firmas nueva y legacy.

        Orden de intento:
        1) ``train_step(epoch, hparams, y=y)``
        2) ``train_step(epoch, hparams)``
        3) ``train_step(epoch)``

        :param epoch: Época actual.
        :param hparams: Hparams normalizados.
        :param y: Target opcional (si alguna estrategia lo usa).
        :return: metrics o (loss, metrics).
        """
        try:
            return self.estrategia.train_step(epoch, hparams, y=y)
        except TypeError:
            # Puede que la estrategia no acepte y
            try:
                return self.estrategia.train_step(epoch, hparams)
            except TypeError:
                # Legacy: solo epoch
                return self.estrategia.train_step(epoch)

    def _extract_loss_and_metrics(self, result: TrainStepReturn) -> Tuple[float, Dict[str, Any]]:
        """
        Normaliza el retorno de la estrategia a ``(loss, metrics)``.

        Soporta:
        - result = {"loss": ..., ...}
        - result = (loss, {"accuracy": ...})

        :param result: Retorno de ``train_step``.
        :return: Tupla (loss, metrics).
        """
        if isinstance(result, tuple) and len(result) == 2:
            loss = float(result[0])
            metrics = dict(result[1] or {})
            metrics.setdefault("loss", loss)
            return loss, metrics

        metrics = dict(result or {})
        loss = metrics.get("loss", metrics.get("recon_error", float("nan")))
        try:
            loss_f = float(loss)
        except Exception:
            loss_f = float("nan")
        metrics["loss"] = loss_f
        return loss_f, metrics

    # ----------------------------------------------------------
    # Método principal de entrenamiento
    # ----------------------------------------------------------
    def run(
        self,
        data_ref: str,
        epochs: int,
        hparams: Optional[Dict[str, Any]] = None,
        model_name: str = "rbm",
        y: Any = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento completo y publica eventos de progreso.

        La plantilla devuelve un diccionario con:
        - ``job_id``
        - ``status``: completed/failed
        - ``metrics``: métricas finales (pueden incluir matrices/listas)
        - ``history``: lista por época (solo numéricos para graficación)
        - ``error`` si falló

        :param data_ref: Referencia del dataset (ruta a parquet/feature-pack).
        :param epochs: Número de épocas.
        :param hparams: Hiperparámetros del entrenamiento.
        :param model_name: Nombre lógico del modelo (para UI/logs).
        :param y: Target opcional (si una estrategia lo requiere).
        :return: Payload con estado + métricas + history.
        """

        # ID del job (permitir override desde hparams para que router/controlador lo fije)
        hparams_norm = self._normalize_hparams(hparams or {})
        job_id = hparams_norm.get("job_id") or str(uuid.uuid4())

        # Contenedores de salida
        history: List[Dict[str, Any]] = []
        last_metrics: Dict[str, Any] = {}

        try:
            # -----------------------------------------
            # Preparación de la estrategia (carga de datos/modelo)
            # -----------------------------------------
            self.estrategia.setup(data_ref, hparams_norm)

            # -----------------------------------------
            # Evento: entrenamiento iniciado
            # -----------------------------------------
            emit_training_started(job_id, model_name, hparams_norm)

            # -----------------------------------------
            # Bucle principal de épocas
            # -----------------------------------------
            for epoch in range(1, int(epochs) + 1):
                t0 = time.perf_counter()

                # Ejecuta 1 época (soporta firma nueva/legacy)
                step_result = self._call_train_step(epoch, hparams_norm, y=y)
                loss, metrics = self._extract_loss_and_metrics(step_result)

                last_metrics = dict(metrics or {})

                dt_ms = (time.perf_counter() - t0) * 1000.0

                # Unificar métricas (enriched se manda a UI vía eventos)
                enriched: Dict[str, Any] = dict(last_metrics)
                enriched["time_epoch_ms"] = float(dt_ms)

                # Default: si no hay recon_error, usa loss como aproximación
                enriched.setdefault("recon_error", float(loss))

                # Agregar a history solo lo numérico (para graficar)
                hist_item: Dict[str, Any] = {"epoch": float(epoch), "loss": float(loss)}
                for k, v in enriched.items():
                    if isinstance(v, (int, float)):
                        hist_item[k] = float(v)
                history.append(hist_item)

                # -----------------------------------------
                # Evento: final de época
                # -----------------------------------------
                emit_epoch_end(job_id, epoch, float(loss), enriched)

                # Pequeño retraso opcional para suavizar gráficas (no bloqueante)
                time.sleep(0.01)

            # -----------------------------------------
            # Finalización exitosa
            # -----------------------------------------
            final_loss = float(history[-1]["loss"]) if history else float("nan")
            final_metrics: Dict[str, Any] = dict(last_metrics)
            final_metrics.setdefault("loss_final", final_loss)
            final_metrics.setdefault("recon_error_final", history[-1].get("recon_error", final_loss) if history else final_loss)

            emit_training_completed(job_id, final_metrics)

            return {
                "job_id": job_id,
                "status": "completed",
                "metrics": final_metrics,
                "history": history,
            }

        except Exception as e:
            # -----------------------------------------
            # Fallo del entrenamiento
            # -----------------------------------------
            emit_training_failed(job_id, str(e))
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "history": history,
            }
