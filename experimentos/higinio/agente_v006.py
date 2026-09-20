"""Variante v006: v004 con verificación de cifras contra XBRL."""

from __future__ import annotations

from experimentos.higinio.agente_v004 import crear_constructor as crear_v004
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = (
    "higinio-v006-hibrido-rrf-reescritura-llm-y-verificacion-xbrl"
)


def crear_constructor() -> ConstructorAgente:
    """Añade a la v004 solo el middleware de verificación XBRL."""
    base = crear_v004()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(VerificadorCifrasXBRL(base.corpus),),
    )
