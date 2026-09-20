"""Middleware de verificación de cifras contra los hechos XBRL del corpus."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pandas as pd
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from pydantic import JsonValue

import miax_s2
from taller_nlp import CorpusVariant


TOLERANCIA_XBRL = 0.01
MARCA_VERIFICACION = "VERIFICACIÓN AUTOMÁTICA"
_COLUMNAS_XBRL = frozenset(
    {"ticker", "fiscal_year", "concept", "value", "unit"}
)


class VerificadorCifrasXBRL(AgentMiddleware):
    """Devuelve al modelo una cifra que no aparece en el XBRL declarado."""

    def __init__(
        self,
        corpus: CorpusVariant,
        *,
        tolerancia: float = TOLERANCIA_XBRL,
        marca: str = MARCA_VERIFICACION,
    ) -> None:
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        if isinstance(tolerancia, bool) or tolerancia < 0:
            raise ValueError("tolerancia debe ser un número no negativo.")
        if not isinstance(marca, str) or not marca.strip():
            raise ValueError("marca debe ser texto no vacío.")

        self._corpus = corpus
        self._tolerancia = float(tolerancia)
        self._marca = marca.strip()
        self._xbrl = pd.read_parquet(corpus.ruta_xbrl)
        ausentes = _COLUMNAS_XBRL - frozenset(self._xbrl.columns)
        if ausentes:
            raise ValueError(
                f"Faltan columnas en el XBRL: {sorted(ausentes)}."
            )

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "tolerancia": self._tolerancia,
                "marca": self._marca,
                "maximo_correcciones": 1,
            }
        )

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Contrasta la respuesta estructurada y solicita una corrección."""
        del runtime
        respuesta = state.get("structured_response")
        cifra = getattr(respuesta, "cifra", None)
        if cifra is None:
            return None

        ticker = getattr(respuesta, "ticker", None)
        ejercicio = getattr(respuesta, "ejercicio", None)
        if ticker is None or ejercicio is None:
            return None
        if self._ya_solicito_correccion(state.get("messages", ())):
            return None

        ticker_normalizado = str(ticker).strip().upper()
        if not ticker_normalizado:
            return None
        try:
            ejercicio_normalizado = int(ejercicio)
            cifra_afirmada = float(cifra)
        except (TypeError, ValueError):
            return None

        filas = self._xbrl[
            (self._xbrl["ticker"].astype(str).str.upper() == ticker_normalizado)
            & (
                self._xbrl["fiscal_year"].astype(int)
                == ejercicio_normalizado
            )
        ]
        valores = tuple(float(valor) for valor in filas["value"])
        if any(
            miax_s2.cuadra(cifra_afirmada, valor, self._tolerancia)
            for valor in valores
        ):
            return None

        hechos = self._formatear_hechos(filas)
        mensaje = (
            f"{self._marca}: afirmaste {cifra_afirmada:,.15g} para "
            f"{ticker_normalizado} FY{ejercicio_normalizado}, pero esa cifra "
            f"no coincide con ningún hecho XBRL reportado con una tolerancia "
            f"del {self._tolerancia:.1%}. {hechos} Corrige la cifra usando "
            "get_xbrl_fact o indica que el dato no está en el corpus."
        )
        return {
            "messages": [{"role": "user", "content": mensaje}],
            "jump_to": "model",
        }

    def _ya_solicito_correccion(self, mensajes: object) -> bool:
        if not isinstance(mensajes, (list, tuple)):
            return False
        return any(
            self._marca in self._contenido_mensaje(mensaje)
            for mensaje in mensajes
        )

    @staticmethod
    def _contenido_mensaje(mensaje: object) -> str:
        if isinstance(mensaje, dict):
            contenido = mensaje.get("content", "")
        else:
            contenido = getattr(mensaje, "text", None)
            if contenido is None:
                contenido = getattr(mensaje, "content", "")
        return str(contenido)

    @staticmethod
    def _formatear_hechos(filas: pd.DataFrame) -> str:
        if filas.empty:
            return "No hay hechos XBRL para esa compañía y ejercicio."
        hechos = "; ".join(
            f"{fila.concept}={float(fila.value):,.15g} {fila.unit}"
            for fila in filas.itertuples(index=False)
        )
        return f"Hechos XBRL disponibles: {hechos}."
