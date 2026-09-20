"""Variante v013: v012 con selección dirigida del fragmento citado."""

from __future__ import annotations

from experimentos.higinio.agente_v012 import crear_constructor as crear_v012
from experimentos.higinio.comun import ampliar_prompt, extender_variante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v013-v012-con-seleccion-fragmento"

INSTRUCCIONES_SELECCION_FRAGMENTO = """Selección de evidencia en comparativas:
- En la primera llamada a search_filings, omite el filtro item para que compitan
  los fragmentos de Items 7 y 8. Busca el nombre financiero exacto en inglés e
  incluye ambos valores cuando ya los conozcas.
- Si la pregunta solo pide cómo evolucionó una magnitud, elige preferentemente
  un fragmento que contenga el nombre de la magnitud y los dos valores
  comparados. La cita debe respaldar la afirmación principal, no ser un
  comentario genérico sobre liquidez, gastos o crecimiento.
- Si la primera búsqueda no devuelve esa línea o tabla, usa la única
  reformulación permitida: busca en Item 8 para estados financieros y tablas,
  o en Item 7 cuando la pregunta solicite causas o explicaciones de la dirección.
- Copia como cita la línea más breve que contenga la magnitud y ambos valores,
  junto con el chunk_id exacto del mismo fragmento."""


def crear_constructor() -> ConstructorAgente:
    """Conserva v012 y añade solo criterios de selección de evidencia."""
    base = crear_v012()
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(),
        configuracion=ampliar_prompt(
            base.configuracion,
            INSTRUCCIONES_SELECCION_FRAGMENTO,
        ),
    )
