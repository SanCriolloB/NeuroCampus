"""neurocampus.utils.score_postprocess
=====================================

Postproceso de scores para la pestaña Predicciones (family: score_docente).

Este módulo es la fuente única de verdad para:

- Mapeo de ``score_total_pred`` (0–50) a categoría de riesgo.
- Cálculo de la tasa de confiabilidad basada en evidencia histórica.
- Construcción de datos para el radar proporcional (perfil de indicadores).
- Construcción de datos para el bar chart comparativo (docente vs cohorte).

Diseño
------
Todas las funciones son **puras** (sin I/O, sin efectos secundarios) para
facilitar pruebas unitarias aisladas. El router y el job de batch las llaman
directamente.

Constantes configurables
------------------------
``N_REF``
    Número de encuestas mínimas para que un par alcance confianza plena (1.0).
``RISK_HIGH_MAX``
    Umbral superior para riesgo "high" (score < este valor).
``RISK_MEDIUM_MAX``
    Umbral superior para riesgo "medium" (score <= este valor).
``INDICATOR_NAMES``
    Nombres de las 10 dimensiones del radar, en el mismo orden que
    ``mean_calif_1..mean_calif_10`` en ``pair_matrix.parquet``.
"""

from __future__ import annotations

from typing import Dict, List, Literal


# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

#: Encuestas mínimas para que ``cov = 1.0`` (cobertura completa).
N_REF: int = 15

#: Score máximo exclusivo para riesgo "high".
RISK_HIGH_MAX: float = 30.0

#: Score máximo inclusivo para riesgo "medium" (entre HIGH_MAX y MEDIUM_MAX).
RISK_MEDIUM_MAX: float = 40.0

#: Nombres de los 10 indicadores del radar, en orden de ``mean_calif_1..10``.
INDICATOR_NAMES: List[str] = [
    "Planificación",
    "Metodología",
    "Claridad",
    "Evaluación",
    "Materiales",
    "Interacción",
    "Retroalimentación",
    "Innovación",
    "Puntualidad",
    "Disponibilidad",
]


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def compute_risk(score: float) -> Literal["low", "medium", "high"]:
    """Mapea un score 0–50 a la categoría de riesgo correspondiente.

    Args:
        score: Valor predicho por el modelo, en escala 0–50 (ya clippeado).

    Returns:
        ``"high"`` si score < :data:`RISK_HIGH_MAX`,
        ``"medium"`` si score <= :data:`RISK_MEDIUM_MAX`,
        ``"low"`` en caso contrario.
    """
    if score < RISK_HIGH_MAX:
        return "high"
    if score <= RISK_MEDIUM_MAX:
        return "medium"
    return "low"


def compute_confidence(*, n_par: int, std_score: float) -> float:
    """Calcula la tasa de confiabilidad (0–1) para un par docente–materia.

    Combina dos factores derivados del historial del par en ``pair_matrix``:

    - **Cobertura** (``cov``): proporción de encuestas disponibles respecto al
      mínimo de referencia :data:`N_REF`. Satura en 1.0.
    - **Estabilidad** (``stability``): penaliza la variabilidad histórica del
      score. Un ``std_score`` alto implica predicciones menos confiables.
      La desviación se normaliza sobre 25.0 (mitad del rango 0–50).

    Fórmula: ``confidence = cov × stability``.

    Args:
        n_par: Número de encuestas históricas para este par específico
            (campo ``n_par`` en ``pair_matrix.parquet``).
        std_score: Desviación estándar del score histórico para este par
            (campo ``std_score_total_0_50`` en ``pair_matrix.parquet``).

    Returns:
        Float en [0, 1] redondeado a 4 decimales.

    Note:
        Si ``n_par == 0`` (par "frío", nunca visto), retorna ``0.0``.
    """
    cov = min(n_par / N_REF, 1.0)
    std_norm = min((std_score or 0.0) / 25.0, 1.0)
    stability = 1.0 - std_norm
    return round(float(cov * stability), 4)


def build_radar(
    *,
    calif_means: List[float],
    score_total_pred: float,
    mean_score_total: float,
) -> List[Dict]:
    """Construye los puntos del radar de indicadores con ajuste proporcional.

    La serie **"Promedio Actual"** usa promedios históricos por pregunta
    (``mean_calif_1..10`` del ``pair_matrix``).

    La serie **"Predicción"** se calcula ajustando cada dimensión actual de forma
    proporcional al score predicho vs el score histórico global, preservando la
    forma del perfil del docente.

    Fórmula::

        ratio = global_pred / max(global_actual, 0.01)
        predicted_dim_i = min(actual_dim_i × ratio, 5.0)

    Donde ``global_actual = mean_score_total / 50 × 5`` y
    ``global_pred = score_total_pred / 50 × 5`` (ambos en escala 0–5).

    Args:
        calif_means: Lista de promedios por pregunta en escala 0–5, en el orden
            de :data:`INDICATOR_NAMES`.
        score_total_pred: Score predicho por el modelo (0–50).
        mean_score_total: Score histórico promedio del par (0–50), campo
            ``mean_score_total_0_50`` en ``pair_matrix``.

    Returns:
        Lista de dicts con claves ``indicator``, ``actual``, ``prediccion``.
    """
    global_actual = mean_score_total / 50.0 * 5.0
    global_pred = score_total_pred / 50.0 * 5.0
    ratio = global_pred / max(global_actual, 0.01)

    n = min(len(INDICATOR_NAMES), len(calif_means))
    return [
        {
            "indicator": INDICATOR_NAMES[i],
            "actual": round(float(calif_means[i]), 3),
            "prediccion": round(min(float(calif_means[i]) * ratio, 5.0), 3),
        }
        for i in range(n)
    ]


def build_comparison(
    *,
    calif_means_docente: List[float],
    calif_means_cohorte: List[float],
) -> List[Dict]:
    """Construye los puntos del bar chart comparativo (docente vs cohorte).

    Args:
        calif_means_docente: Promedios por dimensión del par específico
            (escala 0–5), desde ``mean_calif_1..10`` del par.
        calif_means_cohorte: Promedios por dimensión de todos los pares de esa
            materia (promedio de ``mean_calif_1..10`` filtrando por ``materia_key``).

    Returns:
        Lista de dicts con claves ``dimension``, ``docente``, ``cohorte``.
    """
    n = min(
        len(INDICATOR_NAMES),
        len(calif_means_docente),
        len(calif_means_cohorte),
    )
    return [
        {
            "dimension": INDICATOR_NAMES[i],
            "docente": round(float(calif_means_docente[i]), 3),
            "cohorte": round(float(calif_means_cohorte[i]), 3),
        }
        for i in range(n)
    ]
