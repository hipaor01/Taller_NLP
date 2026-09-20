"""Variante v014: v012 con recuperación híbrida BM25-densa."""

from __future__ import annotations

from experimentos.higinio.agente_v002 import crear_retriever_hibrido
from experimentos.higinio.agente_v012 import crear_constructor as crear_v012
from experimentos.higinio.comun import sustituir_retriever
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v014-v012-con-retrieval-hibrido-rrf"


def crear_constructor() -> ConstructorAgente:
    """Conserva v012 y sustituye únicamente su recuperador por BM25 + denso."""
    base = crear_v012()
    retriever = crear_retriever_hibrido(base.corpus)
    return sustituir_retriever(NOMBRE_VARIANTE, base, retriever)
