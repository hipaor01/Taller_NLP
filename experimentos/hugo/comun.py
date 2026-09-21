"""Piezas comunes a todas las variantes de Hugo.

Aquí se fija el modelo y su tarifa. Todas las versiones de esta carpeta
(v000, v001, ...) usan exactamente la misma configuración de modelo, de modo
que la diferencia medida entre dos versiones se debe solo al cambio que
declara cada una.

Modelo por defecto: Claude Sonnet 5 por la API de Anthropic (decidido el
2026-09-20 tras el diagnóstico sobre 6 preguntas; hasta entonces se usó
Haiku 4.5). Se puede cambiar sin tocar código con la variable de entorno
``HUGO_MODELO``; el modelo tiene que estar en ``PRECIOS`` para que el coste se
mida y no salga como desconocido.

Requisitos:
- ``pip install langchain-anthropic==1.7.1`` (compatible con langchain-core 1.6.1).
- ``ANTHROPIC_API_KEY`` en el entorno. Nunca en un fichero.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from taller_nlp import (
    ConfiguracionAgente,
    ConstructorAgente,
    ControlPeticionesModelo,
    CorpusVariant,
    FabricaHerramientas,
    Retriever,
)

from experimentos.baseline import crear_configuracion_baseline

MODELO_POR_DEFECTO = "anthropic:claude-sonnet-5"

# USD por millón de tokens (entrada, salida). Anthropic no devuelve el coste
# en la respuesta, así que el motor lo estima con estas tarifas.
# Consultado el 2026-09-20 en platform.claude.com/docs/en/about-claude/models.
PRECIOS: dict[str, tuple[float, float]] = {
    "anthropic:claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic:claude-sonnet-5": (2.00, 10.00),
    "anthropic:claude-opus-5": (5.00, 25.00),
    "openrouter:google/gemini-3.8-flash": (0.75, 3.75),
}

# Modelos que rechazan el parámetro temperature (la API responde 400
# "`temperature` is deprecated for this model"). Verificado con Sonnet 5 el
# 2026-09-20. Para ellos el modelo se construye sin temperature, en lugar de
# con temperature=0 como hace el resto del proyecto.
MODELOS_SIN_TEMPERATURA = frozenset({"anthropic:claude-sonnet-5"})

# Tope de tokens de respuesta. Una respuesta estructurada del agente ocupa
# unos cientos de tokens; el tope evita reservas enormes y respuestas
# desbocadas. Es igual en todas las versiones de Hugo.
MAX_TOKENS_SALIDA = 4096


def modelo_configurado() -> str:
    modelo = os.getenv("HUGO_MODELO", MODELO_POR_DEFECTO).strip()
    if modelo not in PRECIOS:
        raise ValueError(
            f"{modelo!r} no tiene tarifa en experimentos/hugo/comun.py. "
            "Añádela antes de evaluar para que el coste se mida."
        )
    return modelo


def crear_configuracion(system_prompt: str | None = None) -> ConfiguracionAgente:
    """La del baseline, cambiando solo el modelo, su tarifa y el tope.

    ``system_prompt`` permite a una versión sustituir el prompt; si se omite
    se conserva el del baseline.
    """
    modelo = modelo_configurado()
    entrada, salida = PRECIOS[modelo]
    datos = crear_configuracion_baseline().model_dump()
    datos.update(
        modelo=modelo,
        precio_entrada_usd_millon_tokens=entrada,
        precio_salida_usd_millon_tokens=salida,
        max_tokens_salida=MAX_TOKENS_SALIDA,
    )
    if system_prompt is not None:
        datos["system_prompt"] = system_prompt
    return ConfiguracionAgente(**datos)


def crear_constructor_hugo(
    nombre: str,
    corpus: CorpusVariant,
    fabrica: FabricaHerramientas,
    *,
    system_prompt: str | None = None,
    middlewares: Sequence[AgentMiddleware] = (),
) -> ConstructorAgente:
    """Ensambla una variante con la evaluación idéntica a la del baseline.

    ``middlewares`` son los guardarraíles propios de la versión (ver
    ``guardrails.py``). El motor los ejecuta antes de los límites de llamadas
    del proyecto, sin tocar código común.
    """
    configuracion = crear_configuracion(system_prompt)
    control = ControlPeticionesModelo.desde_configuracion(configuracion)
    # evaluar.py fija esta variable para que una evaluación interrumpida
    # (Ctrl+C, corte de red, fallo del proceso) se reanude sin repetir ni
    # volver a pagar las preguntas ya evaluadas.
    ruta_progreso = os.getenv("HUGO_RUTA_PROGRESO") or None
    return ConstructorAgente(
        nombre=nombre,
        configuracion=configuracion,
        corpus=corpus,
        fabrica_herramientas=fabrica,
        middlewares=tuple(middlewares),
        modelo=_modelo_sin_temperatura(configuracion, control),
        control_peticiones=control,
        # Mismos parámetros de evaluación que el baseline del grupo.
        k_retrieval=5,
        tolerancia_absoluta=0.0,
        tolerancia_relativa=0.01,
        numero_esperado=None,
        minimo_comparativas=0,
        ruta_progreso=ruta_progreso,
    )


def _modelo_sin_temperatura(
    configuracion: ConfiguracionAgente,
    control: ControlPeticionesModelo,
) -> BaseChatModel | None:
    """Modelo construido aquí solo si rechaza temperature; si no, None.

    None deja que el motor del proyecto cree el modelo como siempre (con
    temperature=0). ``configuracion.temperatura`` sigue valiendo 0 en el
    manifiesto, pero para estos modelos no se envía.
    """
    if configuracion.modelo not in MODELOS_SIN_TEMPERATURA:
        return None
    opciones = {
        "timeout": configuracion.timeout_s,
        "max_tokens": configuracion.max_tokens_salida,
    }
    if control.limitador is not None:
        opciones["rate_limiter"] = control.limitador
    return init_chat_model(configuracion.modelo, **opciones)


__all__ = [
    "MAX_TOKENS_SALIDA",
    "MODELO_POR_DEFECTO",
    "MODELOS_SIN_TEMPERATURA",
    "PRECIOS",
    "Retriever",
    "crear_configuracion",
    "crear_constructor_hugo",
    "modelo_configurado",
]
