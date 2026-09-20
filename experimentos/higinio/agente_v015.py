"""Variante v015: v013 con recuperación híbrida BM25-densa."""

from __future__ import annotations

from experimentos.higinio.agente_v002 import crear_retriever_hibrido
from experimentos.higinio.agente_v013 import crear_constructor as crear_v013
from experimentos.higinio.comun import sustituir_retriever
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v015-v013-con-retrieval-hibrido-rrf"


def crear_constructor() -> ConstructorAgente:
    """Conserva v013 y sustituye únicamente su recuperador por BM25 + denso."""
    base = crear_v013()
    retriever = crear_retriever_hibrido(base.corpus)
    return sustituir_retriever(NOMBRE_VARIANTE, base, retriever)
