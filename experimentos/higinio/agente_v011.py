"""Variante v011: v009 con contrato comparativo tolerante."""

from __future__ import annotations

from experimentos.higinio.agente_v009 import crear_constructor as crear_v009
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.contrato_salida import RespuestaFinancieraTolerante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v011-v009-con-contrato-comparativo-tolerante"


def crear_constructor() -> ConstructorAgente:
    """Conserva v009 y normaliza comparativas sin invalidar la salida."""
    return extender_variante(
        NOMBRE_VARIANTE,
        crear_v009(),
        middlewares=(),
        esquema_respuesta=RespuestaFinancieraTolerante,
    )
