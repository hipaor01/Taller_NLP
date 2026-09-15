"""Creación de modelos de chat con opciones homogéneas entre proveedores."""

from __future__ import annotations

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter


def crear_modelo_chat(
    nombre: str,
    *,
    temperatura: float,
    timeout_s: float,
    max_tokens: int | None = None,
    limitador: BaseRateLimiter | None = None,
) -> BaseChatModel:
    """Inicializa un modelo manteniendo el timeout público expresado en segundos.

    ``langchain-openrouter==0.2.8`` es una excepción entre los proveedores
    soportados por ``init_chat_model``: su argumento ``timeout`` representa
    milisegundos. La conversión queda aislada aquí para que el resto del
    proyecto pueda usar siempre segundos.
    """
    es_openrouter = nombre.partition(":")[0].lower() == "openrouter"
    timeout_proveedor: float | int = (
        max(1, round(timeout_s * 1_000)) if es_openrouter else timeout_s
    )
    opciones: dict[str, Any] = {
        "temperature": temperatura,
        "timeout": timeout_proveedor,
    }
    if es_openrouter:
        # El valor predeterminado de 2 permite una ventana de backoff de unos
        # 300 s en esta versión. Una sola ventana evita esperas desproporcionadas
        # ante errores transitorios sin desactivar el retry del SDK.
        opciones["max_retries"] = 1
    if max_tokens is not None:
        opciones["max_tokens"] = max_tokens
    if limitador is not None:
        opciones["rate_limiter"] = limitador
    return init_chat_model(nombre, **opciones)
