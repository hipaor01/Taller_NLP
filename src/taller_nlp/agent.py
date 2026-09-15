"""Fachada pública común para todas las variantes del agente."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Protocol, final

from .contracts import InformeEvaluacion, RespuestaAgente


class MotorAgente(Protocol):
    """Componente intercambiable que ejecuta una variante del agente."""

    def responder(self, pregunta: str) -> RespuestaAgente:
        """Ejecuta el agente y devuelve su respuesta y trayectoria."""
        ...


class EvaluadorAgente(Protocol):
    """Componente intercambiable que aplica la evaluación compartida."""

    def evaluar(
        self,
        *,
        nombre_agente: str,
        responder: Callable[[str], RespuestaAgente],
        ruta_jsonl: Path,
    ) -> InformeEvaluacion:
        """Evalúa una función de respuesta contra un golden set."""
        ...


@final
class AgenteFinanciero:
    """Fachada estable con motor y evaluador inyectables."""

    __slots__ = ("_nombre", "_motor", "_evaluador")

    def __init__(
        self,
        nombre: str,
        motor: MotorAgente,
        evaluador: EvaluadorAgente,
    ) -> None:
        nombre_normalizado = nombre.strip()
        if not nombre_normalizado:
            raise ValueError("El agente debe tener un nombre no vacío.")
        self._nombre = nombre_normalizado
        self._motor = motor
        self._evaluador = evaluador

    @property
    def nombre(self) -> str:
        return self._nombre

    def responder(self, pregunta: str) -> RespuestaAgente:
        """Responde una pregunta y normaliza errores y latencia total."""
        pregunta_normalizada = pregunta.strip()
        if not pregunta_normalizada:
            raise ValueError("La pregunta no puede estar vacía.")

        inicio = perf_counter()
        try:
            respuesta = self._motor.responder(pregunta_normalizada)
        except Exception as exc:
            latencia_ms = (perf_counter() - inicio) * 1_000
            return RespuestaAgente(
                respuesta="",
                fuente="ninguna",
                latencia_ms=latencia_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(respuesta, RespuestaAgente):
            raise TypeError(
                "MotorAgente.responder() debe devolver una RespuestaAgente."
            )

        latencia_ms = (perf_counter() - inicio) * 1_000
        return respuesta.model_copy(update={"latencia_ms": latencia_ms})

    def evaluar(self, ruta_jsonl: str | Path) -> InformeEvaluacion:
        """Evalúa esta variante mediante el evaluador común inyectado."""
        ruta = Path(ruta_jsonl)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el golden set: {ruta}")

        informe = self._evaluador.evaluar(
            nombre_agente=self.nombre,
            responder=self.responder,
            ruta_jsonl=ruta,
        )
        if not isinstance(informe, InformeEvaluacion):
            raise TypeError(
                "EvaluadorAgente.evaluar() debe devolver un InformeEvaluacion."
            )
        if informe.nombre_agente != self.nombre:
            raise ValueError(
                "El nombre del informe no coincide con el agente evaluado."
            )
        return informe
