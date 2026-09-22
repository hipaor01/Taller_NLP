"""Ciclo de vida de recursos externos usados durante una ejecución."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from threading import RLock
from typing import Any, TypeAlias


FabricaRecursoEjecucion: TypeAlias = Callable[
    [], AbstractContextManager[Any]
]


class GestorRecursosEjecucion:
    """Activa recursos declarados y admite sesiones anidadas de forma segura.

    Las factorías permiten retrasar efectos externos (por ejemplo, arrancar una
    base vectorial) hasta que el agente vaya a responder o evaluar. La primera
    sesión abre los recursos y la última en cerrarse los libera en orden inverso.
    """

    def __init__(
        self,
        fabricas: Sequence[FabricaRecursoEjecucion] = (),
    ) -> None:
        self._fabricas = tuple(fabricas)
        if any(not callable(fabrica) for fabrica in self._fabricas):
            raise TypeError(
                "Cada recurso de ejecución debe ser una factoría callable."
            )
        self._lock = RLock()
        self._profundidad = 0
        self._pila: ExitStack | None = None

    @property
    def fabricas(self) -> tuple[FabricaRecursoEjecucion, ...]:
        return self._fabricas

    @contextmanager
    def sesion(self) -> Iterator[None]:
        """Mantiene disponibles todos los recursos durante el bloque."""
        with self._lock:
            if self._profundidad == 0:
                pila = ExitStack()
                try:
                    for fabrica in self._fabricas:
                        pila.enter_context(fabrica())
                except BaseException:
                    pila.close()
                    raise
                self._pila = pila
            self._profundidad += 1
            try:
                yield
            finally:
                self._profundidad -= 1
                if self._profundidad == 0:
                    assert self._pila is not None
                    pila, self._pila = self._pila, None
                    pila.close()
