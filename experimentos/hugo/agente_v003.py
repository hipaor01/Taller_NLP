"""v003 · Protocolo para las preguntas comparativas.

Único cambio respecto a v002: se añade al system prompt el protocolo que
debe seguir una pregunta que compara dos ejercicios. Ni el modelo, ni el
retriever, ni los límites cambian.

Motivo (evaluación de v002 con Sonnet 5, golden propio, 2026-09-20):
comparativas 2/8. En los 8 casos la cifra era correcta, pero en 6 el agente
respondió solo con dos llamadas a get_xbrl_fact y sin cita, y una comparativa
sin cita no puntúa: el golden espera ``["get_xbrl_fact", "search_filings"]``
y el evaluador exige además que la cita esté literal en el fragmento citado.
Los dos aciertos (gjhh-015 y gjhh-019) son justo los dos casos en los que el
agente buscó además en el MD&A y citó.

El protocolo no menciona ninguna pregunta concreta del golden: describe el
contrato de salida para cualquier comparativa, así que vale igual para las
preguntas ciegas.

Ejecución:
    python -m experimentos.hugo.evaluar agente_v003 golden_set.jsonl v003_propio
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente

from experimentos.baseline import crear_corpus_baseline, crear_fabrica_baseline
from experimentos.hugo.agente_v001 import crear_retriever
from experimentos.hugo.agente_v002 import SYSTEM_PROMPT_V002
from experimentos.hugo.comun import crear_constructor_hugo

NOMBRE = "hugo-v003-protocolo-comparativas"

PROTOCOLO_COMPARATIVAS = """
Preguntas que comparan dos ejercicios (cómo evolucionó, cuánto creció, qué
variación hubo, FY2024 frente a FY2025):
1. Llama a get_xbrl_fact una vez por ejercicio, con el MISMO concepto
   US-GAAP, para tener los dos valores exactos.
2. Llama además a search_filings acotando ticker, fiscal_year al ejercicio
   más reciente e item="7" (el MD&A, donde la dirección comenta la
   variación). Si el concepto es una partida del balance o de la cuenta de
   resultados y no aparece en el MD&A, busca en item="8".
3. Cita una frase del fragmento recuperado que hable de esa magnitud o de su
   variación, y rellena chunk_id con el fragmento del que la has copiado.
4. fuente="ambas": la cifra sale del XBRL y la explicación del texto.
5. cifra: el valor del ejercicio más reciente. La variación va en la prosa.

Una comparativa sin llamada a search_filings y sin cita literal cuenta como
fallo aunque la cifra sea correcta.
"""

SYSTEM_PROMPT_V003 = SYSTEM_PROMPT_V002.rstrip() + "\n" + PROTOCOLO_COMPARATIVAS


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    retriever = crear_retriever(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(
        NOMBRE, corpus, fabrica, system_prompt=SYSTEM_PROMPT_V003
    )
