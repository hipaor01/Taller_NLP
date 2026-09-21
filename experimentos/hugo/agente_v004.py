"""v004 · Middleware propio de comprobación de cifras contra XBRL.

Único cambio respecto a v003: el motor lleva ``ComprobadorCifrasXBRL``
(ver ``guardrails.py``). Es el middleware que pide el enunciado: contrasta la
cifra de la respuesta estructurada con los hechos XBRL del corpus y devuelve
al modelo el desajuste para que lo corrija, una sola vez por pregunta.

Qué añade sobre una comprobación de valor a secas:
- exige que la cifra salga de una llamada a get_xbrl_fact de esta ejecución,
  porque el enunciado cuenta como fallo acertar una cifra leyéndola del texto;
- lee el valor del parquet a partir de los argumentos de la llamada, no de su
  salida en texto, que redondea (get_xbrl_fact imprime el beneficio por acción
  de AAPL FY2024, 6,08 USD/shares, como «6»);
- distingue el caso «no hay hechos de esa compañía y ejercicio en el corpus»,
  donde recuerda que la respuesta correcta es fuente='ninguna' con cifra=null:
  ese es el caso de las preguntas ciegas sin respuesta;
- acepta sin corrección las cifras derivadas de hechos consultados (una
  diferencia, una variación porcentual, un margen).

Ejecución:
    python -m experimentos.hugo.evaluar agente_v004 golden_set.jsonl v004_propio
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente

from experimentos.baseline import crear_corpus_baseline, crear_fabrica_baseline
from experimentos.hugo.agente_v001 import crear_retriever
from experimentos.hugo.agente_v003 import SYSTEM_PROMPT_V003
from experimentos.hugo.comun import crear_constructor_hugo
from experimentos.hugo.guardrails import ComprobadorCifrasXBRL

NOMBRE = "hugo-v004-middleware-xbrl"


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    retriever = crear_retriever(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(
        NOMBRE,
        corpus,
        fabrica,
        system_prompt=SYSTEM_PROMPT_V003,
        middlewares=(ComprobadorCifrasXBRL(corpus),),
    )
