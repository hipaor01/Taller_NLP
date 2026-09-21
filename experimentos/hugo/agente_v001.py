"""v001 · Filtro de metadatos en search_filings.

Único cambio respecto a v000: el retriever FAISS aplica los filtros
``ticker``, ``fiscal_year`` e ``item`` que el agente ya pasa a
``search_filings``. En el baseline esos argumentos se aceptaban por contrato
pero no restringían el ranking denso.

Modelo, prompt, docstrings, límites y evaluación: los de ``comun.py``,
idénticos a v000.

Ejecución:
    python -m agente --variante experimentos.hugo.agente_v001 \
        --evaluar golden_set.jsonl \
        --salida experimentos/hugo/resultados/v001_propio.csv
"""

from __future__ import annotations

from taller_nlp import ConstructorAgente, CorpusVariant

from experimentos.baseline import crear_corpus_baseline, crear_fabrica_baseline
from experimentos.hugo.comun import crear_constructor_hugo
from experimentos.hugo.retrievers import RetrieverFaissSeguro

NOMBRE = "hugo-v001-filtro-metadatos"


def crear_retriever(corpus: CorpusVariant) -> RetrieverFaissSeguro:
    """Denso BGE como el baseline, pero respetando los filtros.

    ``RetrieverFaissSeguro`` solo añade un cerrojo a la codificación (ver
    ``retrievers.py``); el ranking es idéntico al de ``RetrieverFaiss``.
    """
    return RetrieverFaissSeguro(
        corpus,
        nombre="faiss-filtro-metadatos",
        aplicar_filtros_metadatos=True,
    )


def crear_constructor() -> ConstructorAgente:
    corpus = crear_corpus_baseline()
    retriever = crear_retriever(corpus)
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return crear_constructor_hugo(NOMBRE, corpus, fabrica)
