"""Contratos para construir variantes troceadas del corpus."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .component import ComponenteConfigurable


_EXPRESION_COMPONENTE_ID = r"[A-Z0-9.-]+"
_EXPRESION_CHUNK_ID = (
    rf"{_EXPRESION_COMPONENTE_ID}-\d{{4}}-"
    rf"{_EXPRESION_COMPONENTE_ID}-\d{{4,}}"
)


class CorteFragmento(BaseModel):
    """Decisión de corte producida por una estrategia de troceado."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inicio_car: int = Field(ge=0)
    fin_car: int = Field(gt=0)
    n_tokens: int = Field(gt=0)
    contiene_tabla: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.fin_car <= self.inicio_car:
            raise ValueError("fin_car debe ser posterior a inicio_car.")


class FragmentoCorpus(BaseModel):
    """Chunk final persistible, independiente del método de recuperación."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    fiscal_year: int = Field(gt=0)
    item: str = Field(min_length=1)
    posicion: int = Field(ge=0)
    texto: str = Field(min_length=1)
    n_tokens: int = Field(gt=0)
    contiene_tabla: bool
    inicio_car: int = Field(ge=0)
    fin_car: int = Field(gt=0)

    @field_validator("chunk_id")
    @classmethod
    def normalizar_chunk_id(cls, valor: str) -> str:
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El identificador no puede estar vacío.")
        if "[" in normalizado or "]" in normalizado or "\n" in normalizado:
            raise ValueError("El identificador contiene caracteres no válidos.")
        if not re.fullmatch(_EXPRESION_CHUNK_ID, normalizado):
            raise ValueError(
                "chunk_id no sigue el formato TICKER-FY-ITEM-POSICION."
            )
        return normalizado

    @field_validator("ticker", "item")
    @classmethod
    def normalizar_componente(cls, valor: str) -> str:
        normalizado = valor.strip().upper()
        if not re.fullmatch(_EXPRESION_COMPONENTE_ID, normalizado):
            raise ValueError("El componente contiene caracteres no válidos.")
        return normalizado

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("El fragmento no puede estar vacío.")
        return valor

    def model_post_init(self, __context: object) -> None:
        if self.fin_car <= self.inicio_car:
            raise ValueError("fin_car debe ser posterior a inicio_car.")
        if len(self.texto) != self.fin_car - self.inicio_car:
            raise ValueError(
                "La longitud del texto no coincide con inicio_car y fin_car."
            )
        chunk_id_esperado = (
            f"{self.ticker}-{self.fiscal_year}-{self.item}-"
            f"{self.posicion:04d}"
        )
        if self.chunk_id != chunk_id_esperado:
            raise ValueError(
                f"chunk_id no coincide con sus metadatos: se esperaba "
                f"{chunk_id_esperado}."
            )


class TroceadorCorpus(ComponenteConfigurable, ABC):
    """Base para estrategias intercambiables de troceado.

    Las subclases deciden exclusivamente los intervalos en
    ``_calcular_cortes``. Esta clase crea el texto literal y los identificadores
    estables, y rechaza cortes desordenados, duplicados o fuera de la sección.
    Se permite el solapamiento entre fragmentos de forma intencionada.
    """

    def trocear(
        self,
        *,
        ticker: str,
        fiscal_year: int,
        item: str,
        texto: str,
    ) -> tuple[FragmentoCorpus, ...]:
        """Convierte una sección en chunks validados y con ID estable."""
        ticker = self._normalizar_componente("ticker", ticker)
        item = self._normalizar_componente("item", item)
        if (
            isinstance(fiscal_year, bool)
            or not isinstance(fiscal_year, int)
            or fiscal_year <= 0
        ):
            raise ValueError("fiscal_year debe ser un entero mayor que cero.")
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("texto debe ser una sección no vacía.")

        cortes = tuple(
            self._calcular_cortes(
                texto=texto,
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
            )
        )
        self._validar_cortes(cortes, texto=texto)
        return tuple(
            FragmentoCorpus(
                chunk_id=self._crear_chunk_id(
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    item=item,
                    posicion=posicion,
                ),
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
                posicion=posicion,
                texto=texto[corte.inicio_car : corte.fin_car],
                n_tokens=corte.n_tokens,
                contiene_tabla=corte.contiene_tabla,
                inicio_car=corte.inicio_car,
                fin_car=corte.fin_car,
            )
            for posicion, corte in enumerate(cortes)
        )

    @abstractmethod
    def _calcular_cortes(
        self,
        *,
        texto: str,
        ticker: str,
        fiscal_year: int,
        item: str,
    ) -> Iterable[CorteFragmento]:
        """Devuelve cortes ordenados; puede hacer que se solapen."""

    @staticmethod
    def _normalizar_componente(nombre: str, valor: str) -> str:
        if not isinstance(valor, str) or not valor.strip():
            raise ValueError(f"{nombre} debe ser texto no vacío.")
        normalizado = valor.strip().upper()
        if not re.fullmatch(_EXPRESION_COMPONENTE_ID, normalizado):
            raise ValueError(f"{nombre} contiene caracteres no válidos.")
        return normalizado

    @staticmethod
    def _validar_cortes(
        cortes: tuple[CorteFragmento, ...], *, texto: str
    ) -> None:
        if not cortes:
            raise ValueError(
                "El troceador no produjo ningún corte para una sección no vacía."
            )
        if any(not isinstance(corte, CorteFragmento) for corte in cortes):
            raise TypeError("_calcular_cortes debe devolver CorteFragmento.")

        intervalos = [(corte.inicio_car, corte.fin_car) for corte in cortes]
        if len(intervalos) != len(set(intervalos)):
            raise ValueError("El troceador produjo cortes duplicados.")
        if any(fin > len(texto) for _, fin in intervalos):
            raise ValueError("El troceador produjo un corte fuera de la sección.")
        inicios = [inicio for inicio, _ in intervalos]
        if inicios != sorted(inicios):
            raise ValueError("Los cortes deben estar ordenados por inicio_car.")

        fin_cubierto = 0
        for inicio, fin in intervalos:
            if inicio > fin_cubierto and texto[fin_cubierto:inicio].strip():
                raise ValueError(
                    "El troceador dejó texto no vacío fuera de los chunks."
                )
            fin_cubierto = max(fin_cubierto, fin)
        if texto[fin_cubierto:].strip():
            raise ValueError(
                "El troceador dejó texto no vacío fuera de los chunks."
            )

    @staticmethod
    def _crear_chunk_id(
        *, ticker: str, fiscal_year: int, item: str, posicion: int
    ) -> str:
        return f"{ticker}-{fiscal_year}-{item}-{posicion:04d}"
