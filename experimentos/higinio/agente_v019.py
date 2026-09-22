"""Variante v019: v018 con fallback conservador de salida estructurada."""

from __future__ import annotations

from experimentos.higinio.agente_v018 import crear_constructor as crear_v018
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.recuperador_salida import RecuperadorSalidaEstructurada
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v019-v018-con-fallback-estructurado"


def crear_constructor() -> ConstructorAgente:
    """Conserva v018 y recupera solo respuestas textuales inequívocas."""
    base = crear_v018()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(
            RecuperadorSalidaEstructurada(
                base.corpus,
                base.esquema_respuesta,
            ),
        ),
    )
