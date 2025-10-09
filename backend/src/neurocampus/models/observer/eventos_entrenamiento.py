from typing import Dict, Any
from ...observability.bus_eventos import BUS

TRAINING_STARTED = "training.started"
TRAINING_EPOCH_END = "training.epoch_end"
TRAINING_COMPLETED = "training.completed"
TRAINING_FAILED = "training.failed"

def emit_training_started(job_id: str, modelo: str, params: Dict[str, Any]):
    BUS.publish(TRAINING_STARTED, {"correlation_id": job_id, "model": modelo, "params": params})

def emit_epoch_end(job_id: str, epoch: int, loss: float, metrics: Dict[str, float]):
    BUS.publish(TRAINING_EPOCH_END, {"correlation_id": job_id, "epoch": epoch, "loss": loss, "metrics": metrics})

def emit_training_completed(job_id: str, metrics: Dict[str, float]):
    BUS.publish(TRAINING_COMPLETED, {"correlation_id": job_id, "final_metrics": metrics})

def emit_training_failed(job_id: str, error_msg: str):
    BUS.publish(TRAINING_FAILED, {"correlation_id": job_id, "error": error_msg})