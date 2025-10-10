from typing import Dict, Any, Protocol, Tuple, List, Optional
from ..observer.eventos_entrenamiento import (
    emit_training_started, emit_epoch_end, emit_training_completed, emit_training_failed
)
import uuid
import time

class EstrategiaEntrenamiento(Protocol):
    """Contrato para estrategias: RBM general / restringida."""
    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None: ...
    def train_step(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Devuelve (loss, metrics) por epoch"""
        ...

class PlantillaEntrenamiento:
    """Template Method para orquestar entrenamiento con observabilidad y acumulación de history[]."""
    def __init__(self, estrategia: EstrategiaEntrenamiento):
        self.estrategia = estrategia

    def _normalize_hparams(self, hparams: Dict[str, Any]) -> Dict[str, Any]:
        # Normaliza claves a minúsculas por robustez y consistencia de contratos
        return {str(k).lower(): v for k, v in (hparams or {}).items()}

    def run(
        self,
        data_ref: str,
        epochs: int,
        hparams: Dict[str, Any],
        model_name: str = "rbm"
    ) -> Dict[str, Any]:
        # job_id desde hparams o generado
        job_id = (hparams or {}).get("job_id") or str(uuid.uuid4())

        # Normaliza hparams antes de usarlos/empezar
        hparams = self._normalize_hparams(hparams or {})

        # Contenedores de estado/observabilidad
        history: List[Dict[str, float]] = []
        last_metrics: Dict[str, float] = {}

        try:
            # Preparación de la estrategia (carga de datos, inicialización de pesos, etc.)
            self.estrategia.setup(data_ref, hparams)

            # Evento de inicio
            emit_training_started(job_id, model_name, hparams)

            # Bucle de entrenamiento por época
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()

                # Paso de entrenamiento delegado en la estrategia
                loss, metrics = self.estrategia.train_step(epoch)
                last_metrics = dict(metrics or {})

                # Enriquecer métricas con tiempo por época (ms)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                metrics_enriched: Dict[str, float] = dict(last_metrics)
                metrics_enriched["time_epoch_ms"] = float(dt_ms)

                # Asegurar que loss esté también en metrics (útil para UI)
                # y que recon_error exista (por compatibilidad: usar loss si no viene)
                metrics_enriched.setdefault("loss", float(loss))
                metrics_enriched.setdefault("recon_error", float(loss))

                # Guardar en history (solo valores numéricos)
                hist_item: Dict[str, float] = {"epoch": float(epoch), "loss": float(loss)}
                for k, v in metrics_enriched.items():
                    if isinstance(v, (int, float)) and k not in ("epoch",):
                        hist_item[k] = float(v)
                history.append(hist_item)

                # Evento al finalizar cada epoch (enviar métricas enriquecidas)
                emit_epoch_end(job_id, epoch, float(loss), metrics_enriched)

                # Simula tiempo de cómputo (opcional)
                time.sleep(0.01)

            # Métricas finales (ej. recon_error_final basado en la última época)
            final_loss = float(history[-1]["loss"]) if history else float("nan")
            final_metrics = dict(last_metrics)
            final_metrics["recon_error_final"] = float(history[-1].get("recon_error", final_loss)) if history else final_loss

            # Evento de completado
            emit_training_completed(job_id, final_metrics)

            # Respuesta con history[] acumulado
            return {
                "job_id": job_id,
                "status": "completed",
                "metrics": final_metrics,
                "history": history,
            }

        except Exception as e:
            # Evento de fallo
            emit_training_failed(job_id, str(e))
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "history": history,  # devuelve progreso parcial si lo hay
            }
