"""Extracción y estimación homogéneas del uso de modelos de chat."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage


def extraer_uso_modelo(
    mensajes: Sequence[BaseMessage],
) -> tuple[int | None, int | None, float | None]:
    """Suma tokens de entrada, salida y coste informados por el proveedor."""
    entrada = 0
    salida = 0
    coste = 0.0
    hay_tokens = False
    hay_coste = False
    for mensaje in mensajes:
        if not isinstance(mensaje, AIMessage):
            continue
        uso = mensaje.usage_metadata
        if uso:
            entrada += int(uso.get("input_tokens", 0))
            salida += int(uso.get("output_tokens", 0))
            hay_tokens = True
        coste_mensaje = mensaje.response_metadata.get("cost")
        if coste_mensaje is not None:
            coste += float(coste_mensaje)
            hay_coste = True
    return (
        entrada if hay_tokens else None,
        salida if hay_tokens else None,
        coste if hay_coste else None,
    )


def estimar_coste_modelo(
    tokens_entrada: int | None,
    tokens_salida: int | None,
    precio_entrada_usd_millon_tokens: float | None,
    precio_salida_usd_millon_tokens: float | None,
) -> float | None:
    """Estima coste solo cuando están disponibles todos los operandos."""
    if (
        tokens_entrada is None
        or tokens_salida is None
        or precio_entrada_usd_millon_tokens is None
        or precio_salida_usd_millon_tokens is None
    ):
        return None
    return (
        tokens_entrada * precio_entrada_usd_millon_tokens
        + tokens_salida * precio_salida_usd_millon_tokens
    ) / 1_000_000
