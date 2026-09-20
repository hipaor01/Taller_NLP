"""Variante v007: v005 con reparación determinista de citas."""

from __future__ import annotations

from experimentos.higinio.agente_v005 import crear_constructor as crear_v005
from experimentos.higinio.verificador_citas import VerificadorCitas
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v007-v005-con-verificacion-citas"


def crear_constructor() -> ConstructorAgente:
    """Añade a la v005 solo el reparador determinista de citas."""
    base = crear_v005()
    evaluador = base.evaluador
    return ConstructorAgente(
        nombre=NOMBRE_VARIANTE,
        configuracion=base.configuracion,
        corpus=base.corpus,
        fabrica_herramientas=base.fabrica_herramientas,
        middlewares=(*base.middlewares, VerificadorCitas(base.corpus)),
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
