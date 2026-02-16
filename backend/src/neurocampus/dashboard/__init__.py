"""neurocampus.dashboard

Paquete de utilidades del Dashboard.

Este paquete concentra la lógica de lectura/consulta del histórico para los
endpoints ``/dashboard/*``. El objetivo es que los routers permanezcan delgados
(HTTP/serialización) y la lógica de datos viva aquí.

Reglas de negocio clave
-----------------------
- El Dashboard **solo** consulta histórico (nunca datasets individuales):
  - ``historico/unificado.parquet`` (processed histórico)
  - ``historico/unificado_labeled.parquet`` (labeled histórico)

Los módulos dentro de este paquete se implementan por fases según el plan de
trabajo del Dashboard.
"""
