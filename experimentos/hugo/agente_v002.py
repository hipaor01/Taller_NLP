"""v002 · Formato de la respuesta estructurada.

Único cambio respecto a v001: se añaden al system prompt del baseline tres
reglas sobre CÓMO rellenar los campos de ``RespuestaFinanciera``. No cambia
qué herramientas usa el agente ni cómo busca.

Motivo (diagnóstico de v001 con ``evaluar.py``, 2026-09-20):
- Las numéricas gjhh-007, 010 y 011 tenían el valor correcto de XBRL pero
  expresado en millones (cifra=12914, unidad="millones USD"). El contrato del
  enunciado usa la cifra completa en la unidad base (60922000000.0, "USD").
- En comparativas, la cifra era a veces la variación (gjhh-014: 22.885) y
  otras el último valor. Los dos golden sets, el propio y el oficial, esperan
  el valor del ejercicio más reciente.
- En gjhh-003 la cita era la frase correcta pero con apóstrofo recto (') en
  lugar del tipográfico (’) del informe, así que no aparece literal en el
  fragmento. El 25 % de los fragmentos del corpus lleva ’.

Estas reglas salen del esquema del enunciado, no de preguntas concretas del
golden set: aplican igual a las preguntas ciegas. El ejemplo numérico del
prompt es inventado a propósito, para no filtrar respuestas del golden set.

Ejecución:
    python -m experimentos.hugo.evaluar agente_v002 golden_set.jsonl v002_propio
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente

from experimentos.baseline import (
    SYSTEM_PROMPT,
    crear_corpus_baseline,
    crear_fabrica_baseline,
)
from experimentos.hugo.agente_v001 import crear_retriever
from experimentos.hugo.comun import crear_constructor_hugo

NOMBRE = "hugo-v002-formato-respuesta"

REGLAS_FORMATO = """
Formato de la respuesta estructurada:
- cifra: el número COMPLETO en la unidad base, exactamente como lo devuelve
  get_xbrl_fact. Ejemplo: si devuelve 1,234,000,000 USD, entonces
  cifra=1234000000 y unidad="USD". Nunca en millones ni en miles. En la prosa
  de `respuesta` sí puedes escribir "1.234 millones de dólares".
- unidad: la unidad tal cual la da get_xbrl_fact ("USD", "USD/shares").
- Si la pregunta compara dos ejercicios, cifra es el valor del ejercicio MÁS
  RECIENTE. La variación (absoluta o en %) va solo en la prosa de `respuesta`.
- cita: copia literal, carácter a carácter, de una frase del fragmento citado.
  No la traduzcas ni la resumas y conserva la puntuación original, incluidos
  los apóstrofos y comillas tipográficos (’ “ ”).
"""

SYSTEM_PROMPT_V002 = SYSTEM_PROMPT.rstrip() + "\n" + REGLAS_FORMATO


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    retriever = crear_retriever(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(
        NOMBRE, corpus, fabrica, system_prompt=SYSTEM_PROMPT_V002
    )
