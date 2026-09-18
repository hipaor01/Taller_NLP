"""Recolección aislada de telemetría para llamadas auxiliares a modelos."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

from .contracts import LlamadaModeloAuxiliar


class CapturaTelemetriaAuxiliar:
    """Acumulador de solo lectura entregado al dueño de una ejecución."""

    __slots__ = ("_llamadas", "_lock")

    def __init__(self) -> None:
        self._llamadas: list[LlamadaModeloAuxiliar] = []
        self._lock = Lock()

    @property
    def llamadas(self) -> tuple[LlamadaModeloAuxiliar, ...]:
        with self._lock:
            return tuple(self._llamadas)

    def _registrar(self, llamada: LlamadaModeloAuxiliar) -> None:
        with self._lock:
            self._llamadas.append(llamada)


class RegistroTelemetriaAuxiliar:
    """Asocia llamadas auxiliares con la ejecución activa del agente."""

    def __init__(self) -> None:
        self._captura_actual: ContextVar[CapturaTelemetriaAuxiliar | None] = (
            ContextVar(
                f"captura_telemetria_auxiliar_{id(self)}",
                default=None,
            )
        )

    @contextmanager
    def capturar(self) -> Iterator[CapturaTelemetriaAuxiliar]:
        """Abre una captura aislada para la ejecución del contexto actual."""
        captura = CapturaTelemetriaAuxiliar()
        token = self._captura_actual.set(captura)
        try:
            yield captura
        finally:
            self._captura_actual.reset(token)

    def registrar(self, llamada: LlamadaModeloAuxiliar) -> bool:
        """Registra una llamada; devuelve False si no hay ejecución activa."""
        if not isinstance(llamada, LlamadaModeloAuxiliar):
            raise TypeError("llamada debe ser una LlamadaModeloAuxiliar.")
        captura = self._captura_actual.get()
        if captura is None:
            return False
        captura._registrar(llamada)
        return True
