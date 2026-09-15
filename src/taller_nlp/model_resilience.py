"""Control compartido de ritmo y reintentos de llamadas a modelos."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from .config import ConfiguracionAgente


T = TypeVar("T")


class RateLimitAgotadoError(RuntimeError):
    """Se agotaron los reintentos de una petición limitada por el proveedor."""


def es_error_rate_limit(error: BaseException) -> bool:
    """Reconoce un HTTP 429 incluso si otra librería lo ha encapsulado."""
    visitados: set[int] = set()
    actual: BaseException | None = error
    while actual is not None and id(actual) not in visitados:
        visitados.add(id(actual))
        if type(actual).__name__ == "TooManyRequestsResponseError":
            return True
        if getattr(actual, "status_code", None) == 429:
            return True
        respuesta = getattr(actual, "raw_response", None)
        if respuesta is None:
            respuesta = getattr(actual, "response", None)
        if getattr(respuesta, "status_code", None) == 429:
            return True
        actual = actual.__cause__ or actual.__context__
    return False


def texto_indica_rate_limit(texto: str | None) -> bool:
    """Detecta la normalización textual que realiza la fachada pública."""
    if not texto:
        return False
    normalizado = texto.casefold()
    return any(
        marca in normalizado
        for marca in (
            "ratelimitagotadoerror",
            "toomanyrequestsresponseerror",
            "rate limit exceeded",
            "status code: 429",
            "status_code=429",
        )
    )


class ControlPeticionesModelo:
    """Combina un token bucket compartido con backoff selectivo para 429."""

    def __init__(
        self,
        *,
        solicitudes_por_minuto: float | None,
        max_reintentos: int = 3,
        espera_inicial_s: float = 5,
        factor_espera: float = 2,
        espera_maxima_s: float = 20,
        jitter_s: float = 0.5,
        dormir: Callable[[float], None] = time.sleep,
        generar_jitter: Callable[[float, float], float] = random.uniform,
        al_reintentar: (
            Callable[[int, float, BaseException], None] | None
        ) = None,
    ) -> None:
        if solicitudes_por_minuto is not None and solicitudes_por_minuto <= 0:
            raise ValueError("solicitudes_por_minuto debe ser mayor que cero.")
        if max_reintentos < 0:
            raise ValueError("max_reintentos no puede ser negativo.")
        if espera_inicial_s < 0 or espera_maxima_s < 0 or jitter_s < 0:
            raise ValueError("Las esperas y el jitter no pueden ser negativos.")
        if factor_espera < 1:
            raise ValueError("factor_espera debe ser mayor o igual que uno.")
        if espera_inicial_s > espera_maxima_s:
            raise ValueError("La espera inicial no puede superar la máxima.")

        self._max_reintentos = max_reintentos
        self._espera_inicial_s = espera_inicial_s
        self._factor_espera = factor_espera
        self._espera_maxima_s = espera_maxima_s
        self._jitter_s = jitter_s
        self._dormir = dormir
        self._generar_jitter = generar_jitter
        self._al_reintentar = al_reintentar
        self._limitador: BaseRateLimiter | None = None
        if solicitudes_por_minuto is not None:
            self._limitador = InMemoryRateLimiter(
                requests_per_second=solicitudes_por_minuto / 60,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        self._parametros = {
            "solicitudes_por_minuto": solicitudes_por_minuto,
            "max_reintentos": max_reintentos,
            "espera_inicial_s": espera_inicial_s,
            "factor_espera": factor_espera,
            "espera_maxima_s": espera_maxima_s,
            "jitter_s": jitter_s,
        }

    @classmethod
    def desde_configuracion(
        cls, configuracion: ConfiguracionAgente
    ) -> "ControlPeticionesModelo":
        return cls(
            solicitudes_por_minuto=(
                configuracion.solicitudes_modelo_por_minuto
            ),
            max_reintentos=configuracion.max_reintentos_rate_limit,
            espera_inicial_s=configuracion.espera_inicial_rate_limit_s,
            factor_espera=configuracion.factor_espera_rate_limit,
            espera_maxima_s=configuracion.espera_maxima_rate_limit_s,
            jitter_s=configuracion.jitter_rate_limit_s,
        )

    @property
    def limitador(self) -> BaseRateLimiter | None:
        """Token bucket que debe inyectarse en todos los modelos compartidos."""
        return self._limitador

    @property
    def parametros(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._parametros))

    def ejecutar(self, operacion: Callable[[], T]) -> T:
        """Reintenta solo respuestas 429; otros errores se propagan intactos."""
        for numero_reintento in range(self._max_reintentos + 1):
            try:
                return operacion()
            except Exception as exc:
                if not es_error_rate_limit(exc):
                    raise
                if numero_reintento == self._max_reintentos:
                    raise RateLimitAgotadoError(
                        "OpenRouter siguió devolviendo HTTP 429 tras "
                        f"{self._max_reintentos} reintentos."
                    ) from exc
                espera = self._calcular_espera(numero_reintento, exc)
                if self._al_reintentar is not None:
                    self._al_reintentar(numero_reintento + 1, espera, exc)
                self._dormir(espera)
        raise AssertionError("Bucle de reintentos inalcanzable.")

    def _calcular_espera(
        self, numero_reintento: int, error: BaseException
    ) -> float:
        exponencial = self._espera_inicial_s * (
            self._factor_espera**numero_reintento
        )
        indicada = self._extraer_retry_after(error)
        base = max(exponencial, indicada or 0)
        jitter = self._generar_jitter(0, self._jitter_s)
        return min(self._espera_maxima_s, base + jitter)

    @staticmethod
    def _extraer_retry_after(error: BaseException) -> float | None:
        actual: BaseException | None = error
        visitados: set[int] = set()
        while actual is not None and id(actual) not in visitados:
            visitados.add(id(actual))
            respuesta = getattr(actual, "raw_response", None)
            if respuesta is None:
                respuesta = getattr(actual, "response", None)
            headers = getattr(respuesta, "headers", {})
            valor = (
                headers.get("retry-after")
                if hasattr(headers, "get")
                else None
            )
            try:
                if valor is not None:
                    return max(0, float(valor))
            except (TypeError, ValueError):
                pass
            actual = actual.__cause__ or actual.__context__
        return None


class ReintentoRateLimitMiddleware(AgentMiddleware):
    """Aplica al modelo del agente la política común de reintentos 429."""

    def __init__(self, control: ControlPeticionesModelo) -> None:
        self._control = control

    @property
    def parametros(self) -> Mapping[str, Any]:
        return self._control.parametros

    def wrap_model_call(self, request: Any, handler: Callable[[Any], T]) -> T:
        return self._control.ejecutar(lambda: handler(request))
