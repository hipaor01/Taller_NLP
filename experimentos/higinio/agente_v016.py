"""Variante v016: v015 con Gemma 4 26B A4B mediante OpenRouter."""

from __future__ import annotations

from experimentos.higinio.agente_v015 import crear_constructor as crear_v015
from experimentos.higinio.comun import extender_variante
from taller_nlp import ConstructorAgente


NOMBRE_VARIANTE = "higinio-v016-v015-con-gemma-4-26b-a4b"
MODELO = "openrouter:google/gemma-4-26b-a4b-it"
PRECIO_ENTRADA_USD_MILLON_TOKENS = 0.042
PRECIO_SALIDA_USD_MILLON_TOKENS = 0.22


def crear_constructor() -> ConstructorAgente:
    """Conserva v015 y sustituye únicamente el modelo y sus tarifas."""
    base = crear_v015()
    configuracion = base.configuracion.model_copy(
        update={
            "modelo": MODELO,
            "precio_entrada_usd_millon_tokens": (
                PRECIO_ENTRADA_USD_MILLON_TOKENS
            ),
            "precio_salida_usd_millon_tokens": (
                PRECIO_SALIDA_USD_MILLON_TOKENS
            ),
        }
    )
    return extender_variante(
        NOMBRE_VARIANTE,
        base,
        middlewares=(),
        configuracion=configuracion,
    )
