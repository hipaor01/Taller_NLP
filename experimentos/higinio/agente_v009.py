"""Variante v009: v008 con reparación determinista de citas."""

from __future__ import annotations

from experimentos.higinio.agente_v008 import crear_constructor as crear_v008
from experimentos.higinio.comun import extender_variante
from experimentos.higinio.verificador_citas import VerificadorCitas
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v009-v008-con-verificacion-citas"


def crear_constructor() -> ConstructorAgente:
    """Combina el guardrail comparativo de v008 con el reparador de citas."""
    base = crear_v008()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(VerificadorCitas(base.corpus),),
    )
