"""Fallback conservador para respuestas finales sin salida estructurada."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import JsonValue, ValidationError

from experimentos.higinio.traza import extraer_chunks_recuperados_busqueda
from experimentos.higinio.verificador_citas import VerificadorCitas
from taller_nlp import CorpusVariant, RespuestaFinanciera
from taller_nlp.citas import cargar_fragmentos, cita_esta_respaldada


_JSON_CERCADO = re.compile(
    r"\A```(?:json)?\s*(\{.*\})\s*```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class RecuperadorSalidaEstructurada(AgentMiddleware):
    """Reconstruye solo salidas textuales respaldadas inequívocamente."""

    def __init__(
        self,
        corpus: CorpusVariant,
        esquema_respuesta: type[RespuestaFinanciera],
    ) -> None:
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        if not isinstance(esquema_respuesta, type) or not issubclass(
            esquema_respuesta,
            RespuestaFinanciera,
        ):
            raise TypeError(
                "esquema_respuesta debe heredar de RespuestaFinanciera."
            )
        self._esquema_respuesta = esquema_respuesta
        self._chunks = cargar_fragmentos(corpus.ruta_chunks)

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        return MappingProxyType(
            {
                "solo_texto_final_no_estructurado": True,
                "excluye_trazas_xbrl": True,
                "requiere_evidencia_literal_univoca": True,
                "permite_json_textual_valido": True,
            }
        )

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Recupera una respuesta sin iniciar nuevas llamadas al modelo."""
        del runtime
        if isinstance(state.get("structured_response"), RespuestaFinanciera):
            return None

        mensajes = state.get("messages", ())
        mensaje_final = self._ultimo_mensaje_asistente(mensajes)
        if mensaje_final is None or self._tiene_llamadas(mensaje_final):
            return None

        recuperados = extraer_chunks_recuperados_busqueda(mensajes)
        if not recuperados or self._uso_xbrl(mensajes):
            return None

        for candidato in self._candidatos_json(mensaje_final):
            respuesta_json = self._recuperar_json_textual(
                candidato,
                recuperados,
            )
            if respuesta_json is not None:
                return {"structured_response": respuesta_json}

        texto = self._contenido_mensaje(mensaje_final).strip()
        if not texto:
            return None

        evidencia = self._seleccionar_evidencia(
            mensajes,
            texto,
            recuperados,
        )
        if evidencia is None:
            return None
        chunk_id, linea = evidencia
        fragmento = self._chunks[chunk_id]
        respuesta = self._esquema_respuesta(
            respuesta=texto,
            cifra=None,
            unidad=None,
            ticker=fragmento.ticker,
            ejercicio=fragmento.fiscal_year,
            fuente="texto",
            cita=linea,
            chunk_id=chunk_id,
        )
        return {"structured_response": respuesta}

    def _recuperar_json_textual(
        self,
        texto: str,
        recuperados: tuple[str, ...],
    ) -> RespuestaFinanciera | None:
        candidato = texto
        if cercado := _JSON_CERCADO.fullmatch(texto):
            candidato = cercado.group(1)
        try:
            contenido = json.loads(candidato)
        except json.JSONDecodeError:
            return None
        if not isinstance(contenido, dict):
            return None
        try:
            respuesta = self._esquema_respuesta.model_validate(contenido)
        except ValidationError:
            return None
        if respuesta.fuente != "texto":
            return None
        if not respuesta.respuesta.strip():
            return None
        if respuesta.chunk_id not in recuperados or respuesta.cita is None:
            return None
        fragmento = self._chunks.get(respuesta.chunk_id)
        if fragmento is None or not cita_esta_respaldada(
            respuesta.cita,
            fragmento.texto,
        ):
            return None
        return respuesta

    def _seleccionar_evidencia(
        self,
        mensajes: object,
        texto_final: str,
        recuperados: tuple[str, ...],
    ) -> tuple[str, str] | None:
        senales = (texto_final, *self._consultas_busqueda(mensajes))
        candidatas: dict[
            tuple[str, str],
            tuple[int, int, float, float],
        ] = {}
        for chunk_id in recuperados:
            fragmento = self._chunks.get(chunk_id)
            if fragmento is None:
                continue
            for linea in (
                linea.strip()
                for linea in fragmento.texto.splitlines()
                if linea.strip()
            ):
                puntuaciones = tuple(
                    puntuacion
                    for senal in senales
                    if (
                        puntuacion := VerificadorCitas.puntuar_linea(
                            senal,
                            linea,
                        )
                    )
                    is not None
                )
                if puntuaciones:
                    candidatas[(chunk_id, linea)] = max(puntuaciones)
        if not candidatas:
            return None
        mejor_puntuacion = max(candidatas.values())
        mejores = tuple(
            evidencia
            for evidencia, puntuacion in candidatas.items()
            if puntuacion == mejor_puntuacion
        )
        return mejores[0] if len(mejores) == 1 else None

    @staticmethod
    def _ultimo_mensaje_asistente(mensajes: object) -> object | None:
        if not isinstance(mensajes, (list, tuple)):
            return None
        for mensaje in reversed(mensajes):
            if isinstance(mensaje, AIMessage):
                return mensaje
            if isinstance(mensaje, dict) and mensaje.get("role") in {
                "assistant",
                "ai",
            }:
                return mensaje
        return None

    @staticmethod
    def _contenido_mensaje(mensaje: object) -> str:
        if isinstance(mensaje, dict):
            return str(mensaje.get("content", ""))
        texto = getattr(mensaje, "text", None)
        if texto is not None:
            return str(texto)
        return str(getattr(mensaje, "content", ""))

    @staticmethod
    def _tiene_llamadas(mensaje: object) -> bool:
        if isinstance(mensaje, AIMessage):
            return bool(mensaje.tool_calls)
        if isinstance(mensaje, dict):
            return bool(mensaje.get("tool_calls"))
        return False

    @classmethod
    def _candidatos_json(cls, mensaje: object) -> tuple[str, ...]:
        candidatos = []
        contenido = cls._contenido_mensaje(mensaje).strip()
        if contenido:
            candidatos.append(contenido)

        if isinstance(mensaje, AIMessage):
            invalidas = mensaje.invalid_tool_calls
            adicionales = mensaje.additional_kwargs
        elif isinstance(mensaje, dict):
            invalidas = mensaje.get("invalid_tool_calls", ())
            adicionales = mensaje.get("additional_kwargs", {})
        else:
            return tuple(candidatos)

        for llamada in invalidas:
            if not isinstance(llamada, Mapping):
                continue
            cls._anadir_argumentos_json(candidatos, llamada.get("args"))

        if isinstance(adicionales, Mapping):
            for llamada in adicionales.get("tool_calls", ()):
                if not isinstance(llamada, Mapping):
                    continue
                funcion = llamada.get("function")
                if not isinstance(funcion, Mapping):
                    continue
                cls._anadir_argumentos_json(
                    candidatos,
                    funcion.get("arguments"),
                )
        return tuple(dict.fromkeys(candidatos))

    @staticmethod
    def _anadir_argumentos_json(
        candidatos: list[str],
        argumentos: object,
    ) -> None:
        if isinstance(argumentos, str) and argumentos.strip():
            candidatos.append(argumentos.strip())
        elif isinstance(argumentos, Mapping):
            candidatos.append(json.dumps(dict(argumentos)))

    @staticmethod
    def _iterar_llamadas(mensajes: object) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(mensajes, (list, tuple)):
            return ()
        llamadas: list[Mapping[str, Any]] = []
        for mensaje in mensajes:
            if isinstance(mensaje, AIMessage):
                candidatas = mensaje.tool_calls
            elif isinstance(mensaje, dict):
                candidatas = mensaje.get("tool_calls", ())
            else:
                continue
            llamadas.extend(
                llamada
                for llamada in candidatas
                if isinstance(llamada, Mapping)
            )
        return tuple(llamadas)

    @classmethod
    def _uso_xbrl(cls, mensajes: object) -> bool:
        return any(
            llamada.get("name") == "get_xbrl_fact"
            for llamada in cls._iterar_llamadas(mensajes)
        )

    @classmethod
    def _consultas_busqueda(cls, mensajes: object) -> tuple[str, ...]:
        consultas = []
        for llamada in cls._iterar_llamadas(mensajes):
            if llamada.get("name") != "search_filings":
                continue
            argumentos = llamada.get("args")
            if not isinstance(argumentos, Mapping):
                continue
            consulta = argumentos.get("query")
            if isinstance(consulta, str) and consulta.strip():
                consultas.append(consulta.strip())
        return tuple(dict.fromkeys(consultas))
