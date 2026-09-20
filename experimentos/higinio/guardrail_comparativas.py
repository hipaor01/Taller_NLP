"""Guardrail para exigir evidencia textual en respuestas comparativas."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import JsonValue

from experimentos.higinio.traza import extraer_chunks_recuperados_busqueda
from taller_nlp import RespuestaFinanciera


MARCA_GUARDRAIL_COMPARATIVO = "GUARDRAIL COMPARATIVO"
_PERIODO = re.compile(r"\b(?:fy\s*)?\d{4}\b")
_RELACION_PERIODOS = re.compile(
    r"\b(?:entre|between|desde|from|de)\s+(?:fy\s*)?\d{4}\s+"
    r"(?:y|e|and|hasta|to|a)\s+(?:fy\s*)?\d{4}\b"
)
_VERSUS_PERIODOS = re.compile(
    r"\b(?:fy\s*)?\d{4}\s+(?:vs\.?|versus)\s+(?:fy\s*)?\d{4}\b"
)
_ACCION_COMPARATIVA = re.compile(
    r"\b(?:compar\w*|evolucion\w*|cambi\w*|variacion\w*|"
    r"aument\w*|disminu\w*|crec\w*)\b"
)


class GuardrailComparativas(AgentMiddleware):
    """Solicita una única corrección si una comparativa carece de evidencia."""

    def __init__(
        self,
        *,
        marca: str = MARCA_GUARDRAIL_COMPARATIVO,
    ) -> None:
        if not isinstance(marca, str) or not marca.strip():
            raise ValueError("marca debe ser texto no vacío.")
        self._marca = marca.strip()

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "deteccion": "periodos_explicitamente_comparados",
                "requiere_search_filings": True,
                "requiere_cita_recuperada": True,
                "maximo_correcciones": 1,
                "marca": self._marca,
            }
        )

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Devuelve al modelo una comparativa sin búsqueda o cita recuperada."""
        del runtime
        respuesta = state.get("structured_response")
        if not isinstance(respuesta, RespuestaFinanciera):
            return None

        mensajes = state.get("messages", ())
        if not self._es_comparativa(mensajes):
            return None
        if self._ya_solicito_correccion(mensajes):
            return None

        recuperados = extraer_chunks_recuperados_busqueda(mensajes)
        cita = respuesta.cita
        cita_valida = (
            bool(cita and cita.strip())
            and respuesta.chunk_id in recuperados
        )
        if recuperados and cita_valida:
            return None

        motivo = (
            "todavía no hay una llamada exitosa a search_filings que haya "
            "devuelto fragmentos"
            if not recuperados
            else (
                "la respuesta no incluye una cita literal y un chunk_id "
                "perteneciente a los resultados de search_filings"
            )
        )
        mensaje = (
            f"{self._marca}: esta pregunta compara dos ejercicios y {motivo}. "
            "Antes de responder de nuevo, usa search_filings para recuperar "
            "la explicación textual pertinente del 10-K. Conserva las cifras "
            "obtenidas mediante get_xbrl_fact e incluye una cita literal y su "
            "chunk_id recuperado en la respuesta final."
        )
        return {
            "messages": [{"role": "user", "content": mensaje}],
            "jump_to": "model",
        }

    def _es_comparativa(self, mensajes: object) -> bool:
        pregunta = self._pregunta_original(mensajes)
        if pregunta is None:
            return False
        normalizada = self._normalizar(pregunta)
        if _RELACION_PERIODOS.search(normalizada):
            return True
        if _VERSUS_PERIODOS.search(normalizada):
            return True
        return (
            len(_PERIODO.findall(normalizada)) >= 2
            and _ACCION_COMPARATIVA.search(normalizada) is not None
        )

    def _pregunta_original(self, mensajes: object) -> str | None:
        if not isinstance(mensajes, (list, tuple)):
            return None
        for mensaje in mensajes:
            contenido = self._contenido_mensaje(mensaje)
            if self._marca in contenido:
                continue
            if isinstance(mensaje, HumanMessage):
                return contenido
            if isinstance(mensaje, dict) and mensaje.get("role") in {
                "human",
                "user",
            }:
                return contenido
        return None

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
    def _normalizar(texto: str) -> str:
        sin_acentos = "".join(
            caracter
            for caracter in unicodedata.normalize("NFKD", texto)
            if not unicodedata.combining(caracter)
        )
        return " ".join(sin_acentos.casefold().split())
