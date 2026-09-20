"""Variante v005: v001 con verificación de cifras contra XBRL."""

from __future__ import annotations

from experimentos.baseline import crear_corpus_baseline
from experimentos.higinio.agente_v001 import crear_retriever_con_filtros
from experimentos.higinio.comun import ensamblar_variante_retrieval
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v005-filtros-y-verificacion-xbrl"


def crear_constructor() -> ConstructorAgente:
    """Añade a la v001 solo el middleware de verificación XBRL."""
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_con_filtros(corpus)
    verificador = VerificadorCifrasXBRL(corpus)
    return ensamblar_variante_retrieval(
        NOMBRE_VARIANTE,
        corpus,
        retriever,
        middlewares=(verificador,),
    )
