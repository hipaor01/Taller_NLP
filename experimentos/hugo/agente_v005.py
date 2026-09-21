"""v005 · Reparación determinista de la cita.

Único cambio respecto a v004: se añade ``ReparadorCitas`` (ver
``guardrails.py``), que no habla con el modelo. Cuando la cita no está
literal en el fragmento citado, busca esa misma frase —tolerando espacios,
mayúsculas, apóstrofos y comillas tipográficos y elipsis— en el fragmento
citado y en los que devolvió search_filings, y la sustituye por el texto
literal del corpus. Si no la encuentra, deja la respuesta como está: no
inventa citas ni sustituye una frase por otra parecida.

Motivo (evaluación de v002 con Sonnet 5, golden propio, 2026-09-20): la tasa
de cita era 57 %. Los fallos no eran de recuperación sino de copiado: el
modelo escribe apóstrofo recto (') donde el informe lleva el tipográfico (’),
que aparece en el 25 % de los fragmentos del corpus, o añade comillas y
elipsis. El evaluador compara la cita como subcadena literal del fragmento,
normalizando solo espacios y mayúsculas.

Por qué determinista y no otra regla en el prompt: v002 ya pide copiar
carácter a carácter y aun así el modelo normaliza la tipografía. Reparar sin
llamar al modelo no gasta ninguna de las 6 iteraciones.

Ejecución:
    python -m experimentos.hugo.evaluar agente_v005 golden_set.jsonl v005_propio
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente

from experimentos.baseline import crear_corpus_baseline, crear_fabrica_baseline
from experimentos.hugo.agente_v001 import crear_retriever
from experimentos.hugo.agente_v003 import SYSTEM_PROMPT_V003
from experimentos.hugo.comun import crear_constructor_hugo
from experimentos.hugo.guardrails import ComprobadorCifrasXBRL, ReparadorCitas

NOMBRE = "hugo-v005-reparador-citas"


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    retriever = crear_retriever(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(
        NOMBRE,
        corpus,
        fabrica,
        system_prompt=SYSTEM_PROMPT_V003,
        # El motor ejecuta los after_model en orden inverso al de la lista:
        # primero el reparador de citas y luego el comprobador de cifras. Si
        # el comprobador pide corrección, la respuesta nueva vuelve a pasar
        # por el reparador, así que la cita definitiva siempre se repara.
        middlewares=(ComprobadorCifrasXBRL(corpus), ReparadorCitas(corpus)),
    )
