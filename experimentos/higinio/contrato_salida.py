"""Contratos de salida experimentales para las variantes de Higinio."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from taller_nlp import RespuestaFinanciera


TipoRespuesta = Literal["extractiva", "numerica", "comparativa"]


class _RespuestaFinancieraComparativa(RespuestaFinanciera):
    """Campos compartidos por los contratos comparativos experimentales."""

    model_config = ConfigDict(extra="forbid")

    tipo_respuesta: TipoRespuesta | None = Field(
        default=None,
        description=(
            "Clasificación de la pregunta: extractiva, numerica o comparativa"
        )
    )
    cifra: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Valor numérico principal. En una comparativa debe ser exactamente "
            "cifra_final, nunca la variación absoluta ni el porcentaje"
        ),
    )
    ejercicio: int | None = Field(
        default=None,
        description=(
            "Ejercicio de cifra. En una comparativa debe ser ejercicio_final"
        ),
    )
    ejercicio_inicial: int | None = Field(
        default=None,
        gt=0,
        description="Primer ejercicio comparado; obligatorio en comparativas",
    )
    cifra_inicial: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Hecho XBRL del primer ejercicio; obligatorio en comparativas"
        ),
    )
    ejercicio_final: int | None = Field(
        default=None,
        gt=0,
        description="Último ejercicio comparado; obligatorio en comparativas",
    )
    cifra_final: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Hecho XBRL del último ejercicio; obligatorio en comparativas"
        ),
    )
    variacion_absoluta: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="cifra_final menos cifra_inicial",
    )
    variacion_porcentual: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description=(
            "Variación porcentual respecto a cifra_inicial; None si es cero"
        ),
    )

    def _normalizar_valores_disponibles(self) -> None:
        """Canoniza el periodo final sin exigir que todos los campos existan."""
        if self.cifra_final is not None:
            self.cifra = self.cifra_final
            if self.ejercicio_final is not None:
                self.ejercicio = self.ejercicio_final
        if self.cifra_inicial is None or self.cifra_final is None:
            return
        self.variacion_absoluta = self.cifra_final - self.cifra_inicial
        self.variacion_porcentual = (
            None
            if self.cifra_inicial == 0
            else self.variacion_absoluta / abs(self.cifra_inicial) * 100
        )


class RespuestaFinancieraEstricta(_RespuestaFinancieraComparativa):
    """Desambigua el valor canónico y rechaza comparativas incompletas."""

    tipo_respuesta: TipoRespuesta = Field(
        description=(
            "Clasificación de la pregunta: extractiva, numerica o comparativa"
        )
    )

    @model_validator(mode="after")
    def normalizar_comparativa(self) -> "RespuestaFinancieraEstricta":
        """Fija la cifra evaluable al valor del ejercicio final."""
        if self.tipo_respuesta != "comparativa":
            return self

        requeridos = {
            "ticker": self.ticker,
            "unidad": self.unidad,
            "ejercicio_inicial": self.ejercicio_inicial,
            "cifra_inicial": self.cifra_inicial,
            "ejercicio_final": self.ejercicio_final,
            "cifra_final": self.cifra_final,
        }
        ausentes = [nombre for nombre, valor in requeridos.items() if valor is None]
        if ausentes:
            raise ValueError(
                "Una comparativa requiere " + ", ".join(ausentes) + "."
            )
        if self.fuente != "ambas":
            raise ValueError("Una comparativa debe declarar fuente='ambas'.")

        assert self.ejercicio_inicial is not None
        assert self.ejercicio_final is not None
        assert self.cifra_inicial is not None
        assert self.cifra_final is not None
        if self.ejercicio_final <= self.ejercicio_inicial:
            raise ValueError(
                "ejercicio_final debe ser posterior a ejercicio_inicial."
            )

        self._normalizar_valores_disponibles()
        return self


class RespuestaFinancieraTolerante(_RespuestaFinancieraComparativa):
    """Normaliza comparativas sin convertir campos ausentes en un error."""

    @model_validator(mode="after")
    def normalizar_comparativa(self) -> "RespuestaFinancieraTolerante":
        """Aplica la normalización solo cuando hay indicios suficientes."""
        comparativa_declarada = self.tipo_respuesta == "comparativa"
        comparativa_inferida = all(
            valor is not None
            for valor in (
                self.ejercicio_inicial,
                self.cifra_inicial,
                self.ejercicio_final,
                self.cifra_final,
            )
        )
        if comparativa_declarada or comparativa_inferida:
            self._normalizar_valores_disponibles()
        return self
