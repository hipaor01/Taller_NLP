"""Variante v006: v004 con verificación de cifras contra XBRL."""

from __future__ import annotations

from experimentos.higinio.agente_v004 import crear_constructor as crear_v004
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = (
    "higinio-v006-hibrido-rrf-reescritura-llm-y-verificacion-xbrl"
)


def crear_constructor() -> ConstructorAgente:
    """Añade a la v004 solo el middleware de verificación XBRL."""
    base = crear_v004()
    evaluador = base.evaluador
    return ConstructorAgente(
        nombre=NOMBRE_VARIANTE,
        configuracion=base.configuracion,
        corpus=base.corpus,
        fabrica_herramientas=base.fabrica_herramientas,
        middlewares=(*base.middlewares, VerificadorCifrasXBRL(base.corpus)),
        modelo=base.modelo,
        control_peticiones=base.control_peticiones,
        telemetria_auxiliar=base.telemetria_auxiliar,
        k_retrieval=evaluador.k_retrieval,
        tolerancia_absoluta=evaluador.tolerancia_absoluta,
        tolerancia_relativa=evaluador.tolerancia_relativa,
        numero_esperado=evaluador.numero_esperado,
        minimo_comparativas=evaluador.minimo_comparativas,
        ruta_progreso=evaluador.ruta_progreso,
    )
