from typing import Dict, Any
from ...observability.bus_eventos import BUS

TRAINING_STARTED = "training.started"
TRAINING_EPOCH_END = "training.epoch_end"
TRAINING_COMPLETED = "training.completed"
TRAINING_PERSISTED = "training.persisted"
TRAINING_FAILED = "training.failed"

def emit_training_started(job_id: str, modelo: str, params: Dict[str, Any]):
    BUS.publish(TRAINING_STARTED, {"correlation_id": job_id, "model": modelo, "params": params})

def emit_epoch_end(job_id: str, epoch: int, loss: float, metrics: Dict[str, float]):
    BUS.publish(TRAINING_EPOCH_END, {"correlation_id": job_id, "epoch": epoch, "loss": loss, "metrics": metrics})

def emit_training_completed(job_id: str, metrics: Dict[str, float]):
    BUS.publish(TRAINING_COMPLETED, {"correlation_id": job_id, "final_metrics": metrics})


def emit_training_persisted(job_id: str, *, run_id: str, artifact_path: str) -> None:
    """Notifica que los artifacts del entrenamiento ya fueron persistidos.

    Este evento se emite desde el router (no desde la plantilla) inmediatamente
    después de ejecutar ``runs_io.save_run``.

    Motivo
    ------
    La plantilla de entrenamiento emite ``training.completed`` *antes* de que el
    router guarde el run en disco.  En ese pequeño intervalo, la UI puede ver
    el job como "completed" pero aún no contar con un ``run_id`` navegable.

    ``training.persisted`` marca el punto en el que:

    - el directorio ``artifacts/runs/<run_id>`` existe, y
    - el cliente puede abrir/inspeccionar ``metrics.json`` y demás artifacts.

    Parameters
    ----------
    job_id:
        Identificador de correlación del job (correspondiente a ``/modelos/estado/{job_id}``).
    run_id:
        Identificador estable del run persistido.
    artifact_path:
        Ruta lógica (relativa) del directorio del run, p.ej. ``artifacts/runs/<run_id>``.
    """

    BUS.publish(
        TRAINING_PERSISTED,
        {
            "correlation_id": job_id,
            "run_id": str(run_id),
            "artifact_path": str(artifact_path),
            "artifact_ready": True,
        },
    )

def emit_training_failed(job_id: str, error_msg: str):
    BUS.publish(TRAINING_FAILED, {"correlation_id": job_id, "error": error_msg})