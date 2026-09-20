"""Variante v010: v009 con contrato comparativo estricto."""

from __future__ import annotations

from experimentos.higinio.agente_v009 import crear_constructor as crear_v009
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.contrato_salida import RespuestaFinancieraEstricta
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v010-v009-con-contrato-comparativo-estricto"


def crear_constructor() -> ConstructorAgente:
    """Conserva v009 y sustituye únicamente su contrato de salida."""
    return extender_variante(
        NOMBRE_VARIANTE,
        crear_v009(),
        middlewares=(),
        esquema_respuesta=RespuestaFinancieraEstricta,
    )
