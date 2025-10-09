from typing import Dict, Any, Protocol, Tuple
from ..observer.eventos_entrenamiento import (
    emit_training_started, emit_epoch_end, emit_training_completed, emit_training_failed
)
import uuid, time

class EstrategiaEntrenamiento(Protocol):
    """Contrato para estrategias: RBM general / restringida."""
    def setup(self, data_ref: str, hparams: Dict[str, Any]) -> None: ...
    def train_step(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Devuelve (loss, metrics) por epoch"""
        ...

class PlantillaEntrenamiento:
    """Template Method para orquestar entrenamiento con observabilidad."""
    def __init__(self, estrategia: EstrategiaEntrenamiento):
        self.estrategia = estrategia

    def run(self, data_ref: str, epochs: int, hparams: Dict[str, Any], model_name: str = "rbm"):
        job_id = hparams.get("job_id") or str(uuid.uuid4())
        try:
            self.estrategia.setup(data_ref, hparams)
            emit_training_started(job_id, model_name, hparams)

            last_metrics = {}
            for epoch in range(1, epochs + 1):
                # Paso de entrenamiento delegado en la estrategia
                loss, metrics = self.estrategia.train_step(epoch)
                last_metrics = metrics

                # Evento al finalizar cada epoch
                emit_epoch_end(job_id, epoch, loss, metrics)
                time.sleep(0.01)  # simula tiempo de cómputo

            emit_training_completed(job_id, last_metrics)
            return {"job_id": job_id, "status": "completed", "metrics": last_metrics}
        except Exception as e:
            emit_training_failed(job_id, str(e))
            return {"job_id": job_id, "status": "failed", "error": str(e)}