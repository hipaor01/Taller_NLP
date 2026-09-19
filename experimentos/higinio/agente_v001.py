"""Variante v001: búsqueda densa con filtrado posterior por metadatos.

Conserva la configuración, el corpus, las herramientas y el protocolo de
evaluación del baseline. La única intervención experimental es activar en el
retriever FAISS los filtros de ticker, ejercicio fiscal e item.
"""

from __future__ import annotations

from experimentos.baseline import (
    crear_corpus_baseline,
)
from experimentos.higinio.comun import ensamblar_variante_retrieval
from taller_nlp import ConstructorAgente, CorpusVariant, RetrieverFaiss


NOMBRE_VARIANTE = "higinio-v001-filtros-metadatos"


def crear_retriever_con_filtros(corpus: CorpusVariant) -> RetrieverFaiss:
    """Busca en todo el índice y filtra el ranking antes de devolver el top-k."""
    return RetrieverFaiss(
        corpus,
        nombre="faiss-denso-con-filtros-metadatos",
        aplicar_filtros_metadatos=True,
    )


def crear_constructor() -> ConstructorAgente:
    """Ensambla la v001 cambiando únicamente el retriever del baseline."""
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_con_filtros(corpus)
    return ensamblar_variante_retrieval(NOMBRE_VARIANTE, corpus, retriever)
