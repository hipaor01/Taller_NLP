"""Implementación del motor del agente sobre LangChain."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ToolCallRequest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from .auxiliary_telemetry import RegistroTelemetriaAuxiliar
from .config import ConfiguracionAgente
from .contracts import (
    LlamadaHerramienta,
    LlamadaModeloAuxiliar,
    RespuestaAgente,
    RespuestaFinanciera,
)
from .corpus import CorpusVariant
from .model_factory import crear_modelo_chat
from .model_usage import estimar_coste_modelo, extraer_uso_modelo
from .model_resilience import (
    ControlPeticionesModelo,
    LimpiarRazonamientoOpenRouterMiddleware,
    ReintentoRateLimitMiddleware,
)
from .retrieval import extraer_chunk_ids_formateados
from .tool_suite import ToolSuite


_MAX_CARACTERES_RESULTADO = 4_000


@dataclass(frozen=True, slots=True)
class EjecucionLangChain:
    """Estado crudo y telemetría de una invocación del grafo."""

    estado: dict[str, Any]
    latencia_ms: float
    coste_usd: float | None
    tokens_entrada: int | None
    tokens_salida: int | None
    llamadas_modelo_auxiliares: tuple[LlamadaModeloAuxiliar, ...] = ()
    coste_modelo_principal_usd: float | None = None
    tokens_entrada_modelo_principal: int | None = None
    tokens_salida_modelo_principal: int | None = None

    @property
    def latencia_s(self) -> float:
        """Latencia total expresada en segundos."""
        return self.latencia_ms / 1_000


class _ToolTimingMiddleware(AgentMiddleware):
    """Anota en cada ToolMessage cuánto tardó su ejecución."""

    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        inicio = perf_counter()
        resultado = handler(request)
        duracion_ms = (perf_counter() - inicio) * 1_000
        if not isinstance(resultado, ToolMessage):
            return resultado

        adicionales = dict(resultado.additional_kwargs)
        adicionales["taller_nlp_duracion_ms"] = duracion_ms
        return resultado.model_copy(update={"additional_kwargs": adicionales})


class MotorLangChain:
    """Motor ReAct configurable construido con ``create_agent``."""

    def __init__(
        self,
        configuracion: ConfiguracionAgente,
        herramientas: ToolSuite,
        corpus: CorpusVariant,
        *,
        middlewares: Sequence[AgentMiddleware] = (),
        modelo: BaseChatModel | None = None,
        control_peticiones: ControlPeticionesModelo | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        telemetria_auxiliar: RegistroTelemetriaAuxiliar | None = None,
    ) -> None:
        self._configuracion = configuracion
        self._herramientas = herramientas
        self._corpus = corpus
        self._middlewares_usuario = tuple(middlewares)
        self._control_peticiones = control_peticiones or (
            ControlPeticionesModelo.desde_configuracion(configuracion)
        )
        self._checkpointer = checkpointer
        self._telemetria_auxiliar = telemetria_auxiliar

        modelo_langchain = modelo or self._crear_modelo(
            configuracion, self._control_peticiones
        )
        middlewares_agente = self._crear_middlewares()
        self._agente = create_agent(
            model=modelo_langchain,
            tools=herramientas.herramientas,
            system_prompt=configuracion.system_prompt,
            middleware=middlewares_agente,
            response_format=ToolStrategy(RespuestaFinanciera),
            name="agente_financiero",
            checkpointer=checkpointer,
        )
        # LangGraph aplica este timeout a cada superstep. Los límites de llamadas
        # acotan además el número máximo de supersteps de una ejecución.
        self._agente.step_timeout = configuracion.timeout_s

    @staticmethod
    def _crear_modelo(
        configuracion: ConfiguracionAgente,
        control_peticiones: ControlPeticionesModelo | None = None,
    ) -> BaseChatModel:
        return crear_modelo_chat(
            configuracion.modelo,
            temperatura=configuracion.temperatura,
            timeout_s=configuracion.timeout_s,
            max_tokens=configuracion.max_tokens_salida,
            limitador=(
                control_peticiones.limitador
                if control_peticiones is not None
                else None
            ),
        )

    def _crear_middlewares(self) -> tuple[AgentMiddleware, ...]:
        limites: list[AgentMiddleware] = [
            ModelCallLimitMiddleware(
                run_limit=self._configuracion.max_iteraciones,
                exit_behavior="end",
            ),
            ToolCallLimitMiddleware(
                run_limit=self._configuracion.max_llamadas_total,
                exit_behavior="end",
            ),
        ]
        limites.extend(
            ToolCallLimitMiddleware(
                tool_name=nombre,
                run_limit=limite,
                exit_behavior="end",
            )
            for nombre, limite in self._configuracion.limites_por_herramienta.items()
        )
        compatibilidad_openrouter: tuple[AgentMiddleware, ...] = ()
        if self._configuracion.modelo.partition(":")[0].lower() == "openrouter":
            compatibilidad_openrouter = (
                LimpiarRazonamientoOpenRouterMiddleware(),
            )
        return (
            _ToolTimingMiddleware(),
            *self._middlewares_usuario,
            *limites,
            *compatibilidad_openrouter,
            ReintentoRateLimitMiddleware(self._control_peticiones),
        )

    @property
    def configuracion(self) -> ConfiguracionAgente:
        return self._configuracion

    @property
    def herramientas(self) -> ToolSuite:
        return self._herramientas

    @property
    def corpus(self) -> CorpusVariant:
        return self._corpus

    @property
    def checkpointer(self) -> BaseCheckpointSaver | None:
        return self._checkpointer

    @property
    def telemetria_auxiliar(self) -> RegistroTelemetriaAuxiliar | None:
        return self._telemetria_auxiliar

    def ejecutar(
        self,
        pregunta: str,
        *,
        thread_id: str | None = None,
    ) -> EjecucionLangChain:
        """Ejecuta el grafo conservando íntegro el estado que devuelve."""
        if thread_id is not None and not isinstance(thread_id, str):
            raise TypeError("thread_id debe ser texto o None.")
        thread_id_normalizado = thread_id.strip() if thread_id is not None else None
        if thread_id is not None and not thread_id_normalizado:
            raise ValueError("thread_id no puede estar vacío.")

        configuracion: dict[str, Any] = {
            "recursion_limit": max(
                50, self._configuracion.max_iteraciones * 8
            ),
            "metadata": {"corpus_variant": self._corpus.nombre},
        }
        if thread_id_normalizado is not None:
            configuracion["configurable"] = {
                "thread_id": thread_id_normalizado,
            }
        elif self._checkpointer is not None:
            # Mantiene independientes las llamadas que no solicitan memoria.
            configuracion["configurable"] = {"thread_id": uuid4().hex}

        inicio = perf_counter()
        telemetria_auxiliar = getattr(self, "_telemetria_auxiliar", None)
        if telemetria_auxiliar is None:
            estado = self._invocar_grafo(pregunta, configuracion)
            llamadas_auxiliares: tuple[LlamadaModeloAuxiliar, ...] = ()
        else:
            with telemetria_auxiliar.capturar() as captura:
                estado = self._invocar_grafo(pregunta, configuracion)
            llamadas_auxiliares = captura.llamadas
        if not isinstance(estado, dict):
            raise TypeError("El grafo debe devolver su estado como un dict.")
        latencia_ms = (perf_counter() - inicio) * 1_000
        mensajes = estado.get("messages", ())
        entrada_principal, salida_principal, coste_principal = self._extraer_uso(
            mensajes
        )
        if coste_principal is None:
            coste_principal = self._estimar_coste(
                entrada_principal,
                salida_principal,
            )
        tokens_entrada = self._sumar_telemetria(
            entrada_principal,
            llamadas_auxiliares,
            "tokens_entrada",
        )
        tokens_salida = self._sumar_telemetria(
            salida_principal,
            llamadas_auxiliares,
            "tokens_salida",
        )
        coste_usd = self._sumar_telemetria(
            coste_principal,
            llamadas_auxiliares,
            "coste_usd",
        )
        return EjecucionLangChain(
            estado=estado,
            latencia_ms=latencia_ms,
            coste_usd=coste_usd,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            llamadas_modelo_auxiliares=llamadas_auxiliares,
            coste_modelo_principal_usd=coste_principal,
            tokens_entrada_modelo_principal=entrada_principal,
            tokens_salida_modelo_principal=salida_principal,
        )

    def _invocar_grafo(
        self,
        pregunta: str,
        configuracion: dict[str, Any],
    ) -> dict[str, Any]:
        return self._agente.invoke(
            {"messages": [{"role": "user", "content": pregunta}]},
            config=configuracion,
        )

    @staticmethod
    def _sumar_telemetria(
        principal: int | float | None,
        llamadas: Sequence[LlamadaModeloAuxiliar],
        campo: str,
    ) -> int | float | None:
        """Suma un campo solo cuando todas las mediciones están disponibles."""
        if not llamadas:
            return principal
        auxiliares = [getattr(llamada, campo) for llamada in llamadas]
        if principal is None or any(valor is None for valor in auxiliares):
            return None
        return principal + sum(auxiliares)

    def _estimar_coste(
        self,
        tokens_entrada: int | None,
        tokens_salida: int | None,
    ) -> float | None:
        """Estima el coste con tarifas declaradas si el proveedor no lo dio."""
        return estimar_coste_modelo(
            tokens_entrada,
            tokens_salida,
            self._configuracion.precio_entrada_usd_millon_tokens,
            self._configuracion.precio_salida_usd_millon_tokens,
        )

    def responder(self, pregunta: str) -> RespuestaAgente:
        """Ejecuta el grafo y transforma su estado al contrato compartido."""
        ejecucion = self.ejecutar(pregunta)
        estado = ejecucion.estado
        mensajes = estado.get("messages", ())
        llamadas = self._extraer_llamadas(mensajes)
        salida = estado.get("structured_response")

        if not isinstance(salida, RespuestaFinanciera):
            texto_final = self._ultimo_texto_asistente(mensajes)
            return RespuestaAgente(
                respuesta=texto_final,
                fuente="ninguna",
                llamadas=llamadas,
                llamadas_modelo_auxiliares=(
                    ejecucion.llamadas_modelo_auxiliares
                ),
                latencia_ms=ejecucion.latencia_ms,
                coste_usd=ejecucion.coste_usd,
                tokens_entrada=ejecucion.tokens_entrada,
                tokens_salida=ejecucion.tokens_salida,
                error="El agente terminó sin una respuesta estructurada.",
            )

        return RespuestaAgente(
            respuesta=salida.respuesta,
            cifra=salida.cifra,
            unidad=salida.unidad,
            fuente=salida.fuente,
            citas=(salida.chunk_id,) if salida.chunk_id else (),
            cita=(
                salida.cita.strip()
                if salida.cita is not None and salida.cita.strip()
                else None
            ),
            llamadas=llamadas,
            llamadas_modelo_auxiliares=ejecucion.llamadas_modelo_auxiliares,
            latencia_ms=ejecucion.latencia_ms,
            coste_usd=ejecucion.coste_usd,
            tokens_entrada=ejecucion.tokens_entrada,
            tokens_salida=ejecucion.tokens_salida,
        )

    @classmethod
    def _extraer_llamadas(
        cls, mensajes: Sequence[BaseMessage]
    ) -> tuple[LlamadaHerramienta, ...]:
        resultados = {
            mensaje.tool_call_id: mensaje
            for mensaje in mensajes
            if isinstance(mensaje, ToolMessage)
        }
        llamadas = []
        for mensaje in mensajes:
            if not isinstance(mensaje, AIMessage):
                continue
            for indice, llamada in enumerate(mensaje.tool_calls):
                nombre = llamada["name"]
                if nombre not in {
                    "list_available",
                    "get_xbrl_fact",
                    "search_filings",
                    "read_section",
                }:
                    continue

                id_llamada = llamada.get("id") or f"{nombre}-{indice}"
                mensaje_tool = resultados.get(id_llamada)
                if mensaje_tool is None:
                    llamadas.append(
                        LlamadaHerramienta(
                            id=id_llamada,
                            nombre=nombre,
                            argumentos=llamada.get("args", {}),
                            duracion_ms=0,
                            error="La llamada no produjo un ToolMessage.",
                        )
                    )
                    continue

                contenido = cls._contenido_como_texto(mensaje_tool.content)
                chunk_ids = (
                    extraer_chunk_ids_formateados(contenido)
                    if nombre == "search_filings"
                    else ()
                )
                duracion_ms = float(
                    mensaje_tool.additional_kwargs.get(
                        "taller_nlp_duracion_ms", 0
                    )
                )
                es_error = mensaje_tool.status == "error"
                llamadas.append(
                    LlamadaHerramienta(
                        id=id_llamada,
                        nombre=nombre,
                        argumentos=llamada.get("args", {}),
                        duracion_ms=duracion_ms,
                        chunk_ids=chunk_ids,
                        resultado=None if es_error else cls._truncar(contenido),
                        error=cls._truncar(contenido) if es_error else None,
                    )
                )
        return tuple(llamadas)

    @staticmethod
    def _extraer_uso(
        mensajes: Sequence[BaseMessage],
    ) -> tuple[int | None, int | None, float | None]:
        return extraer_uso_modelo(mensajes)

    @staticmethod
    def _contenido_como_texto(contenido: Any) -> str:
        if isinstance(contenido, str):
            return contenido
        return json.dumps(contenido, ensure_ascii=False, default=str)

    @staticmethod
    def _ultimo_texto_asistente(mensajes: Sequence[BaseMessage]) -> str:
        for mensaje in reversed(mensajes):
            if isinstance(mensaje, AIMessage):
                return MotorLangChain._contenido_como_texto(mensaje.content)
        return ""

    @staticmethod
    def _truncar(texto: str) -> str:
        if len(texto) <= _MAX_CARACTERES_RESULTADO:
            return texto
        return texto[:_MAX_CARACTERES_RESULTADO] + "… [truncado]"
