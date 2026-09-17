"""Progreso por consola para ejecuciones interactivas del agente."""

from __future__ import annotations

import sys
from time import perf_counter

from langchain.agents.middleware import AgentMiddleware


class ProgresoConsolaMiddleware(AgentMiddleware):
    """Hace visible la fase en curso sin alterar respuestas ni herramientas.

    ``etiqueta`` permite distinguir en la salida el experimento que está
    ejecutándose. Se conserva ``"baseline"`` como valor predeterminado para
    mantener el comportamiento de los scripts existentes.
    """

    def __init__(self, etiqueta: str = "baseline") -> None:
        etiqueta = etiqueta.strip()
        if not etiqueta:
            raise ValueError("La etiqueta de progreso no puede estar vacía.")
        self._etiqueta = etiqueta
        self._numero_llamada_modelo = 0

    @property
    def parametros(self) -> dict[str, object]:
        return {
            "destino": "stderr",
            "etiqueta": self._etiqueta,
            "muestra_argumentos_tools": True,
        }

    def _mostrar(self, mensaje: str) -> None:
        print(
            f"[{self._etiqueta}] {mensaje}",
            file=sys.stderr,
            flush=True,
        )

    def wrap_model_call(self, request, handler):
        self._numero_llamada_modelo += 1
        numero = self._numero_llamada_modelo
        inicio = perf_counter()
        self._mostrar(f"Llamada {numero} al modelo...")
        try:
            resultado = handler(request)
        except Exception as exc:
            self._mostrar(
                f"La llamada {numero} falló: {type(exc).__name__}: {exc}"
            )
            raise
        self._mostrar(
            f"Llamada {numero} completada en {perf_counter() - inicio:.1f} s."
        )
        return resultado

    def wrap_tool_call(self, request, handler):
        nombre = request.tool_call.get("name", "desconocida")
        argumentos = request.tool_call.get("args", {})
        inicio = perf_counter()
        self._mostrar(f"Tool {nombre}({argumentos})...")
        try:
            resultado = handler(request)
        except Exception as exc:
            self._mostrar(
                f"Tool {nombre} falló: {type(exc).__name__}: {exc}"
            )
            raise
        self._mostrar(
            f"Tool {nombre} completada en {perf_counter() - inicio:.1f} s."
        )
        return resultado
