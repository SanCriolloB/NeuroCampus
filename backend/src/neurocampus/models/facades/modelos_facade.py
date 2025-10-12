# backend/src/neurocampus/models/facades/modelos_facade.py
from __future__ import annotations
from typing import Optional, Dict, Any

from ..strategies.metodologia import DatasetResolver, Metodo, MetodoParams

class ModelosFacade:
    """
    Fachada central para orquestar entrenamientos sin exponer detalles internos.
    - Mantiene compatibilidad hacia la API/UI.
    - Permite resolver datasets por 'metodologia' si no se provee 'data_ref'.
    """

    def __init__(self, base_uri: str = "localfs://."):
        self.base_uri = base_uri

    def entrenar(self, modelo: str,
                 data_ref: Optional[str] = None,
                 metodologia: Optional[str] = None,
                 metodologia_params: Optional[Dict[str, Any]] = None,
                 epochs: int = 5,
                 hparams: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Entrena el 'modelo' con los parámetros dados.
        Si 'data_ref' es None y se indica 'metodologia', resuelve el dataset.
        """
        data_meta = None
        if not data_ref and metodologia:
            resolver = DatasetResolver(base_uri=self.base_uri)
            metodo = Metodo(metodologia)  # 'PeriodoActual' | 'Acumulado' | 'Ventana'
            params = MetodoParams(**(metodologia_params or {}))
            data_ref, data_meta = resolver.resolve(metodo, params)

        # TODO: Conectar con tu pipeline real de entrenamiento.
        # Por ejemplo:
        # trainer = self._select_trainer(modelo)
        # result = trainer.train(data_ref=data_ref, epochs=epochs, hparams=hparams or {})
        # return {**result, "data_meta": data_meta}

        # Placeholder seguro para no romper mientras conectas el trainer real:
        return {
            "status": "ok",
            "modelo": modelo,
            "data_ref": data_ref,
            "epochs": epochs,
            "hparams": hparams or {},
            "data_meta": data_meta,
            "msg": "ModelosFacade creado; conecta tu pipeline de entrenamiento en el TODO."
        }

    # def _select_trainer(self, modelo: str):
    #     # TODO: conectar con tu registro de trainers o estrategias
    #     raise NotImplementedError("Seleccionador de trainers no implementado aún.")
