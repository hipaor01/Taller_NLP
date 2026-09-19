"""Variante v003: filtros y reescritura LLM de la consulta de búsqueda."""

from __future__ import annotations

import os

import miax_s2

from experimentos.baseline import (
    crear_configuracion_baseline,
    crear_corpus_baseline,
)
from experimentos.higinio.agente_v001 import crear_retriever_con_filtros
from experimentos.higinio.comun import ensamblar_variante_retrieval
from taller_nlp import (
    ConfiguracionAgente,
    ConstructorAgente,
    ControlPeticionesModelo,
    CorpusVariant,
    ReescritorConsultaLLM,
    RegistroTelemetriaAuxiliar,
    RetrieverConReescritura,
)
from taller_nlp.model_factory import crear_modelo_chat


NOMBRE_VARIANTE = "higinio-v003-reescritura-consulta-llm"
INSTRUCCION = """Reescribe esta pregunta como una consulta de búsqueda para un
índice de informes 10-K en INGLÉS. Usa el vocabulario del propio informe.
Devuelve SOLO la consulta, sin comillas ni explicación."""


def crear_constructor() -> ConstructorAgente:
    """Ensambla la reescritura LLM sobre el denso filtrado de la v001."""
    configuracion = crear_configuracion_baseline()
    control = ControlPeticionesModelo.desde_configuracion(configuracion)
    telemetria = RegistroTelemetriaAuxiliar()
    reescritor = crear_reescritor_consulta(
        configuracion,
        control,
        telemetria,
    )
    corpus = crear_corpus_baseline()
    retriever = crear_retriever_con_reescritura(corpus, reescritor)
    return ensamblar_variante_retrieval(
        NOMBRE_VARIANTE,
        corpus,
        retriever,
        configuracion=configuracion,
        control_peticiones=control,
        telemetria_auxiliar=telemetria,
    )


def crear_reescritor_consulta(
    configuracion: ConfiguracionAgente,
    control: ControlPeticionesModelo,
    telemetria: RegistroTelemetriaAuxiliar,
) -> ReescritorConsultaLLM:
    """Crea el reescritor compartido por las variantes v003 y v004."""
    modelo_reescritor = None
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        try:
            modelo_reescritor = crear_modelo_chat(
                configuracion.modelo,
                temperatura=0,
                timeout_s=configuracion.timeout_s,
                limitador=control.limitador,
            )
        except Exception:
            modelo_reescritor = None

    return ReescritorConsultaLLM(
        nombre_modelo=configuracion.modelo,
        instruccion=INSTRUCCION,
        modelo=modelo_reescritor,
        control_peticiones=control,
        telemetria=telemetria,
        reescrituras_respaldo=miax_s2.REESCRITURAS_RESPALDO,
        precio_entrada_usd_millon_tokens=(
            configuracion.precio_entrada_usd_millon_tokens
        ),
        precio_salida_usd_millon_tokens=(
            configuracion.precio_salida_usd_millon_tokens
        ),
    )


def crear_retriever_con_reescritura(
    corpus: CorpusVariant,
    reescritor: ReescritorConsultaLLM,
) -> RetrieverConReescritura:
    """Compone el reescritor con el retriever filtrado de la v001."""
    return RetrieverConReescritura(
        corpus,
        retriever_base=crear_retriever_con_filtros(corpus),
        reescritor=reescritor,
        nombre="denso-filtrado-con-reescritura-llm",
    )
