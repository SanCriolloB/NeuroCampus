# backend/src/neurocampus/models/templates/plantilla_entrenamiento.py
# -------------------------------------------------------------------
# Plantilla oficial de entrenamiento con soporte completo de eventos
# training.started / training.epoch_end / training.completed / training.failed
# requerida por el frontend para graficar la curva de pérdida en tiempo real.
# -------------------------------------------------------------------

from typing import Dict, Any, Protocol, Tuple, List, Optional
from ..observer.eventos_entrenamiento import (
    emit_training_started,
    emit_epoch_end,
    emit_training_completed,
    emit_training_failed,
)
import uuid
import time


class EstrategiaEntrenamiento(Protocol):
    """
    Contrato para estrategias RBM_general / RBM_restringida / RBM_manual.
    Deben implementar setup() y train_step().
    """

    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None:
        ...

    def train_step(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Devuelve (loss, metrics)"""
        ...


class PlantillaEntrenamiento:
    """
    Template Method para orquestar entrenamiento con eventos compatibles
    con el frontend de NeuroCampus (pestaña Modelos).

    El frontend espera recibir, vía BUS interno:
        - training.started
        - training.epoch_end
        - training.completed
        - training.failed
    """

    def __init__(self, estrategia: EstrategiaEntrenamiento):
        self.estrategia = estrategia

    # ----------------------------------------------------------
    # Normalización homogénea de hiperparámetros
    # ----------------------------------------------------------
    def _normalize_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        return {str(k).lower(): v for k, v in (hparams or {}).items()}

    # ----------------------------------------------------------
    # Método principal de entrenamiento
    # ----------------------------------------------------------
    def run(
        self,
        data_ref: str,
        epochs: int,
        hparams: Dict[str, Any],
        model_name: str = "rbm",
    ) -> Dict[str, Any]:
        """
        Ejecuta entrenamiento completo y publica eventos de progreso.
        Devuelve history y métricas finales para almacenar en memoria (/modelos/estado).
        """

        # ID del job
        job_id = (hparams or {}).get("job_id") or str(uuid.uuid4())

        # Normalizar hiperparámetros
        hparams = self._normalize_hparams(hparams or {})

        # contenedores
        history: List[Dict[str, float]] = []
        last_metrics: Dict[str, float] = {}

        try:
            # -----------------------------------------
            # Preparación de la estrategia (carga de datos inicial)
            # -----------------------------------------
            self.estrategia.setup(data_ref, hparams)

            # -----------------------------------------
            # Evento: entrenamiento iniciado
            # -----------------------------------------
            emit_training_started(job_id, model_name, hparams)

            # -----------------------------------------
            # Bucle principal
            # -----------------------------------------
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()

                # Paso de entrenamiento (la estrategia devuelve loss + métricas)
                loss, metrics = self.estrategia.train_step(epoch)
                last_metrics = dict(metrics or {})

                dt_ms = (time.perf_counter() - t0) * 1000.0

                # Unificar métricas
                enriched = dict(last_metrics)
                enriched["time_epoch_ms"] = float(dt_ms)
                enriched.setdefault("loss", float(loss))
                enriched.setdefault("recon_error", float(loss))

                # Agregar a history
                hist_item = {"epoch": float(epoch), "loss": float(loss)}
                for k, v in enriched.items():
                    if isinstance(v, (int, float)):
                        hist_item[k] = float(v)
                history.append(hist_item)

                # -----------------------------------------
                # Evento: final de época
                # -----------------------------------------
                emit_epoch_end(
                    job_id,
                    epoch,
                    float(loss),
                    enriched,
                )

                # Pequeño retraso opcional para suavizar gráficas
                time.sleep(0.01)

            # -----------------------------------------
            # Finalización
            # -----------------------------------------
            final_loss = float(history[-1]["loss"]) if history else float("nan")
            final_metrics = dict(last_metrics)
            final_metrics["recon_error_final"] = history[-1].get(
                "recon_error", final_loss
            )

            # Evento: completado
            emit_training_completed(job_id, final_metrics)

            # Respuesta con history completo
            return {
                "job_id": job_id,
                "status": "completed",
                "metrics": final_metrics,
                "history": history,
            }

        except Exception as e:
            # Evento: fallo
            emit_training_failed(job_id, str(e))
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "history": history,
            }
