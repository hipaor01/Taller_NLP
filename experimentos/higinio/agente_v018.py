"""Variante v018: v017 con respuesta terminal para cifras XBRL presentes."""

from __future__ import annotations

from experimentos.higinio.agente_v017 import crear_constructor as crear_v017
from experimentos.higinio.comun import ampliar_prompt, extender_variante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v018-v017-con-cifra-xbrl-terminal"

INSTRUCCIONES_CIFRA_XBRL_TERMINAL = """Hechos XBRL presentes — regla prioritaria:
- Una pregunta numérica simple pide una única magnitud de una compañía y un
  ejercicio, sin comparar periodos ni solicitar causas o explicaciones.
- Si get_xbrl_fact devuelve correctamente esa magnitud y ejercicio, considera
  el dato definitivo y responde inmediatamente.
- Después de obtenerlo, NO uses search_filings, read_section, list_available ni
  repitas get_xbrl_fact. Una pregunta numérica simple no necesita evidencia
  textual adicional.
- Devuelve el valor exacto con cifra, unidad, ticker y ejercicio; usa
  fuente="xbrl", cita=null y chunk_id=null.
- Esta regla NO se aplica a comparativas ni a preguntas que pidan causas,
  explicaciones o comentarios de la dirección: conserva para ellas el flujo de
  herramientas y evidencia definido anteriormente."""


def crear_constructor() -> ConstructorAgente:
    """Conserva v017 y añade solo la parada tras un hecho XBRL válido."""
    base = crear_v017()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(),
        configuracion=ampliar_prompt(
            base.configuracion,
            INSTRUCCIONES_CIFRA_XBRL_TERMINAL,
        ),
    )
