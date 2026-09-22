"""v3 · v2 con abstención terminal ante datos XBRL ausentes.

La infraestructura, el modelo, el recuperador y los límites son los de v2. El
único cambio experimental es que una respuesta negativa de ``get_xbrl_fact``
o ``list_available`` pasa a ser definitiva para una cifra solicitada.
"""

from __future__ import annotations

from pathlib import Path

from experimentos.jchulvi import agente_v2
from taller_nlp import ConstructorAgente


NOMBRE = "jchulvi-v3-abstencion-xbrl-terminal"

_REGLA_TEXTO_V2 = """- Busca los importes primero con get_xbrl_fact. Si una partida no figura en
  XBRL, admite una cantidad del informe solo acompañada de su pasaje literal
  y señalando expresamente que procede del texto del 10-K."""

REGLA_AUSENCIA_XBRL = """- Para toda pregunta que solicite una cifra, consulta primero el concepto
  exacto mediante get_xbrl_fact.
- Si la pregunta pide ingresos, ventas o facturación SIN indicar una etiqueta
  XBRL, las etiquetas a comprobar son Revenues y
  RevenueFromContractWithCustomerExcludingAssessedTax. Si una falta, consulta
  la otra antes de declarar ausencia. Usa el dato disponible del ejercicio
  pedido, aunque la primera consulta fallara; también en comparativas.
- Esa selección de etiqueta NO se permite si la pregunta exige un identificador
  XBRL literal: consulta ese concepto y aplica la abstención si falta, aunque
  exista otra etiqueta de ingresos. Las reglas siguientes siguen vigentes.
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
  fuente="ninguna", cita=null y chunk_id=null.
- Aplica la misma salida cuando list_available confirme que la compañía o el
  ejercicio solicitado no están en el corpus.
- Esta regla de abstención afecta a cifras y no impide usar search_filings para
  preguntas extractivas ni para explicar comparativas con hechos XBRL válidos."""


def _crear_instrucciones() -> str:
    """Sustituye la regla incompatible de v2 y detecta cambios silenciosos."""
    if agente_v2.INSTRUCCIONES.count(_REGLA_TEXTO_V2) != 1:
        raise RuntimeError(
            "No se encontró una única regla de respaldo textual en agente_v2."
        )
    return agente_v2.INSTRUCCIONES.replace(
        _REGLA_TEXTO_V2,
        REGLA_AUSENCIA_XBRL,
        1,
    )


INSTRUCCIONES = _crear_instrucciones()


def crear_constructor(
    *,
    ruta_progreso: str | Path | None = None,
) -> ConstructorAgente:
    """Conserva la composición de v2 y sustituye únicamente su prompt."""
    base = agente_v2.crear_constructor(ruta_progreso=ruta_progreso)
    evaluador = base.evaluador
    configuracion = base.configuracion.model_copy(
        update={"system_prompt": INSTRUCCIONES}
    )
    return ConstructorAgente(
        nombre=NOMBRE,
        configuracion=configuracion,
        corpus=base.corpus,
        fabrica_herramientas=base.fabrica_herramientas,
        middlewares=base.middlewares,
        esquema_respuesta=base.esquema_respuesta,
        modelo=base.modelo,
        control_peticiones=base.control_peticiones,
        telemetria_auxiliar=base.telemetria_auxiliar,
        recursos_ejecucion=base.recursos_ejecucion,
        k_retrieval=evaluador.k_retrieval,
        tolerancia_absoluta=evaluador.tolerancia_absoluta,
        tolerancia_relativa=evaluador.tolerancia_relativa,
        numero_esperado=evaluador.numero_esperado,
        minimo_comparativas=evaluador.minimo_comparativas,
        ruta_progreso=evaluador.ruta_progreso,
    )
