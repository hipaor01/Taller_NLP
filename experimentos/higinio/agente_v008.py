"""Variante v008: v005 con guardrail para preguntas comparativas."""

from __future__ import annotations

from experimentos.higinio.agente_v005 import crear_constructor as crear_v005
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.guardrail_comparativas import GuardrailComparativas
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v008-v005-con-guardrail-comparativas"


def crear_constructor() -> ConstructorAgente:
    """Añade a la v005 solo el guardrail de evidencia comparativa."""
    base = crear_v005()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(GuardrailComparativas(),),
    )
