"""Variante v002: filtros de metadatos y retrieval híbrido BM25-denso."""

from __future__ import annotations

import miax_s2

from experimentos.baseline import crear_corpus_baseline
from experimentos.higinio.agente_v001 import crear_retriever_con_filtros
from experimentos.higinio.comun import ensamblar_variante_retrieval
from taller_nlp import ConstructorAgente, CorpusVariant, RetrieverHibridoRRF


NOMBRE_VARIANTE = "higinio-v002-hibrido-bm25-denso-rrf"
KK_RRF = 60


def crear_retriever_hibrido(corpus: CorpusVariant) -> RetrieverHibridoRRF:
    """Añade BM25 + RRF al ranking denso filtrado de la v001."""
    retriever_denso = crear_retriever_con_filtros(corpus)
    bm25, chunks_bm = miax_s2.montar_bm25()
    return RetrieverHibridoRRF(
        corpus,
        retriever_denso=retriever_denso,
        indice_bm25=bm25,
        fragmentos_bm25=chunks_bm,
        tokenizador=miax_s2.tokenizar,
        kk=KK_RRF,
        nombre="hibrido-bm25-denso-con-filtros-rrf",
        nombre_tokenizador="miax_s2.tokenizar",
    )


def crear_constructor() -> ConstructorAgente:
    """Ensambla la v002 cambiando solo el retrieval respecto a la v001."""
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_hibrido(corpus)
    return ensamblar_variante_retrieval(NOMBRE_VARIANTE, corpus, retriever)
