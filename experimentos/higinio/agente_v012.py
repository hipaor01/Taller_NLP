"""Variante v012: v011 con instrucciones comparativas preventivas."""

from __future__ import annotations

from experimentos.higinio.agente_v011 import crear_constructor as crear_v011
from experimentos.higinio.comun import ampliar_prompt, extender_variante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v012-v011-con-prompt-comparativo"

INSTRUCCIONES_COMPARATIVAS = """Para preguntas que comparen dos ejercicios:
1. Consulta cada ejercicio con get_xbrl_fact usando el mismo concepto XBRL.
2. Usa search_filings al menos una vez para recuperar evidencia textual.
3. Si la primera búsqueda devuelve evidencia pertinente, no la repitas.
4. Copia en cita un fragmento literal devuelto por search_filings y usa
   exactamente el chunk_id correspondiente al mismo fragmento.
5. Antes de finalizar, comprueba que cita y chunk_id no estén vacíos.
6. Usa como cifra y ejercicio principales los correspondientes al ejercicio final.
7. Si la primera búsqueda no aporta evidencia suficiente, reformúlala como
   máximo una vez. No inventes una cita ni continúes buscando indefinidamente."""


def crear_constructor() -> ConstructorAgente:
    """Conserva v011 y añade solo instrucciones para comparativas."""
    base = crear_v011()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(),
        configuracion=ampliar_prompt(
            base.configuracion,
            INSTRUCCIONES_COMPARATIVAS,
        ),
    )
