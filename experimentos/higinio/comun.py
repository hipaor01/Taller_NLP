"""Ensamblado compartido por las variantes de retrieval de Higinio."""

from __future__ import annotations

from collections.abc import Sequence

from langchain.agents.middleware import AgentMiddleware

from experimentos.baseline import (
    crear_configuracion_baseline,
    crear_fabrica_baseline,
)
from taller_nlp import (
    ConfiguracionAgente,
    ConstructorAgente,
    ControlPeticionesModelo,
    CorpusVariant,
    FabricaHerramientas,
    RegistroTelemetriaAuxiliar,
    RespuestaFinanciera,
    Retriever,
)


def ampliar_prompt(
    configuracion: ConfiguracionAgente,
    instrucciones: str,
) -> ConfiguracionAgente:
    """Añade instrucciones sin mutar la configuración de la variante base."""
    if not isinstance(instrucciones, str) or not instrucciones.strip():
        raise ValueError("instrucciones debe ser texto no vacío.")
    prompt = f"{configuracion.system_prompt}\n\n{instrucciones.strip()}"
    return configuracion.model_copy(update={"system_prompt": prompt})


def ensamblar_variante_retrieval(
    nombre: str,
    corpus: CorpusVariant,
    retriever: Retriever,
    *,
    configuracion: ConfiguracionAgente | None = None,
    control_peticiones: ControlPeticionesModelo | None = None,
    telemetria_auxiliar: RegistroTelemetriaAuxiliar | None = None,
    middlewares: Sequence[AgentMiddleware] = (),
) -> ConstructorAgente:
    """Conserva todos los parámetros del baseline salvo el retriever."""
    fabrica = crear_fabrica_baseline(corpus, retriever)
    return ConstructorAgente(
        nombre=nombre,
        configuracion=configuracion or crear_configuracion_baseline(),
        corpus=corpus,
        fabrica_herramientas=fabrica,
        middlewares=middlewares,
        control_peticiones=control_peticiones,
        telemetria_auxiliar=telemetria_auxiliar,
        k_retrieval=5,
        tolerancia_absoluta=0.0,
        tolerancia_relativa=0.01,
        numero_esperado=None,
        minimo_comparativas=0,
    )


def extender_variante(
    nombre: str,
    base: ConstructorAgente,
    *,
    middlewares: Sequence[AgentMiddleware],
    configuracion: ConfiguracionAgente | None = None,
    esquema_respuesta: type[RespuestaFinanciera] | None = None,
    fabrica_herramientas: FabricaHerramientas | None = None,
) -> ConstructorAgente:
    """Extiende una variante sin alterar su configuración de evaluación."""
    evaluador = base.evaluador
    return ConstructorAgente(
        nombre=nombre,
        configuracion=configuracion or base.configuracion,
        corpus=base.corpus,
        fabrica_herramientas=(
            fabrica_herramientas or base.fabrica_herramientas
        ),
        middlewares=(*base.middlewares, *middlewares),
        esquema_respuesta=esquema_respuesta or base.esquema_respuesta,
        modelo=base.modelo,
        control_peticiones=base.control_peticiones,
        telemetria_auxiliar=base.telemetria_auxiliar,
        k_retrieval=evaluador.k_retrieval,
        tolerancia_absoluta=evaluador.tolerancia_absoluta,
        tolerancia_relativa=evaluador.tolerancia_relativa,
        numero_esperado=evaluador.numero_esperado,
        minimo_comparativas=evaluador.minimo_comparativas,
        ruta_progreso=evaluador.ruta_progreso,
    )


def sustituir_retriever(
    nombre: str,
    base: ConstructorAgente,
    retriever: Retriever,
) -> ConstructorAgente:
    """Conserva una variante y sustituye solo su recuperador textual."""
    fabrica = FabricaHerramientas(base.corpus, retriever=retriever)
    return extender_variante(
        nombre,
        base,
        middlewares=(),
        fabrica_herramientas=fabrica,
    )
