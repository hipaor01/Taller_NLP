"""Variante v004: híbrido BM25-denso con reescritura LLM previa."""

from __future__ import annotations

from experimentos.baseline import (
    crear_configuracion_baseline,
    crear_corpus_baseline,
)
from experimentos.higinio.agente_v002 import crear_retriever_hibrido
from experimentos.higinio.agente_v003 import crear_reescritor_consulta
from experimentos.higinio.comun import ensamblar_variante_retrieval
from taller_nlp import (
    ConstructorAgente,
    ControlPeticionesModelo,
    CorpusVariant,
    ReescritorConsultaLLM,
    RegistroTelemetriaAuxiliar,
    RetrieverConReescritura,
)


NOMBRE_VARIANTE = "higinio-v004-hibrido-rrf-con-reescritura-llm"


def crear_retriever_combinado(
    corpus: CorpusVariant,
    reescritor: ReescritorConsultaLLM,
) -> RetrieverConReescritura:
    """Reescribe una vez y entrega la misma consulta a denso y BM25."""
    return RetrieverConReescritura(
        corpus,
        retriever_base=crear_retriever_hibrido(corpus),
        reescritor=reescritor,
        nombre="hibrido-rrf-con-reescritura-llm",
    )


def crear_constructor() -> ConstructorAgente:
    """Ensambla la reescritura de v003 sobre el retrieval de v002."""
    configuracion = crear_configuracion_baseline()
    control = ControlPeticionesModelo.desde_configuracion(configuracion)
    telemetria = RegistroTelemetriaAuxiliar()
    reescritor = crear_reescritor_consulta(
        configuracion,
        control,
        telemetria,
    )
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_combinado(corpus, reescritor)
    return ensamblar_variante_retrieval(
        NOMBRE_VARIANTE,
        corpus,
        retriever,
        configuracion=configuracion,
        control_peticiones=control,
        telemetria_auxiliar=telemetria,
    )
