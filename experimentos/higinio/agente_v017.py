"""Variante v017: v015 con abstención terminal ante hechos XBRL ausentes."""

from __future__ import annotations

from experimentos.higinio.agente_v015 import crear_constructor as crear_v015
from experimentos.higinio.comun import ampliar_prompt, extender_variante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v017-v015-con-abstencion-xbrl-terminal"

INSTRUCCIONES_AUSENCIA_XBRL = """Ausencia de hechos XBRL — regla prioritaria:
- Si get_xbrl_fact indica que la compañía no reportó el concepto solicitado en
  ese ejercicio, considera esa respuesta definitiva para la cifra pedida.
- Después de esa confirmación negativa, NO uses search_filings ni read_section
  para buscar la cifra en el texto y NO consultes otros conceptos para
  sustituirla.
- NO calcules ni derives la magnitud mediante identidades contables, sumas,
  restas, porcentajes o conceptos relacionados, aunque el cálculo sea posible.
- NO incluyas en la respuesta cifras de conceptos alternativos. Limítate a
  explicar brevemente que el concepto solicitado no está reportado en el corpus.
- En ese caso responde exactamente con cifra=null, unidad=null,
  fuente="ninguna", cita=null y chunk_id=null. No conserves una unidad si la
  cifra es null.
- Aplica la misma salida cuando list_available confirme que la compañía o el
  ejercicio solicitado no están en el corpus."""


def crear_constructor() -> ConstructorAgente:
    """Conserva v015 y añade solo la política de abstención XBRL."""
    base = crear_v015()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(),
        configuracion=ampliar_prompt(
            base.configuracion,
            INSTRUCCIONES_AUSENCIA_XBRL,
        ),
    )
