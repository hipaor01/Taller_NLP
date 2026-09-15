"""Contrato común para las distintas estrategias de recuperación."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

from pydantic import ConfigDict, Field, JsonValue

from .chunking import FragmentoCorpus, _EXPRESION_CHUNK_ID
from .component import ComponenteConfigurable
from .corpus import CorpusVariant


_PATRON_CABECERA_FRAGMENTO = re.compile(
    rf"(?m)^\[({_EXPRESION_CHUNK_ID})\]\s"
)


def extraer_chunk_ids_formateados(texto: str) -> tuple[str, ...]:
    """Extrae, en orden, los IDs de las cabeceras del retriever."""
    return tuple(_PATRON_CABECERA_FRAGMENTO.findall(texto))


class FragmentoRecuperado(FragmentoCorpus):
    """Fragmento del corpus junto con la información de su recuperación."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    puntuacion: float | None = Field(
        default=None,
        description=(
            "Puntuación propia del algoritmo; no es comparable entre "
            "implementaciones. El ranking lo determina el orden de salida."
        ),
    )
    detalles_puntuacion: dict[str, float] = Field(default_factory=dict)


class Retriever(ComponenteConfigurable, ABC):
    """Base validada para recuperadores dense, léxicos o híbridos.

    Las subclases solo implementan ``_buscar``. Este contrato normaliza la
    consulta y los filtros, valida el top-k y ofrece un formato estable para
    la herramienta ``search_filings``.
    """

    def __init__(
        self,
        nombre: str,
        corpus: CorpusVariant,
        *,
        parametros: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not isinstance(corpus, CorpusVariant):
            raise TypeError("corpus debe ser una instancia de CorpusVariant.")
        super().__init__(nombre, parametros=parametros)
        self._corpus = corpus

    @property
    def corpus(self) -> CorpusVariant:
        return self._corpus

    def buscar(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        item: str | None = None,
        k: int = 5,
    ) -> tuple[FragmentoRecuperado, ...]:
        """Devuelve hasta ``k`` fragmentos ordenados por relevancia."""
        query, ticker, fiscal_year, item, k = self._normalizar_entrada(
            query, ticker, fiscal_year, item, k
        )
        fragmentos = tuple(
            self._buscar(
                query=query,
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
                k=k,
            )
        )
        self._validar_salida(
            fragmentos,
            ticker=ticker,
            fiscal_year=fiscal_year,
            item=item,
            k=k,
        )
        return fragmentos

    def buscar_formateado(
        self,
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        item: str | None = None,
        k: int = 5,
    ) -> str:
        """Recupera y serializa fragmentos para ``search_filings``."""
        return self.formatear(
            self.buscar(
                query=query,
                ticker=ticker,
                fiscal_year=fiscal_year,
                item=item,
                k=k,
            )
        )

    @abstractmethod
    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> Iterable[FragmentoRecuperado]:
        """Implementa el ranking concreto y devuelve los mejores fragmentos."""

    @staticmethod
    def formatear(fragmentos: Sequence[FragmentoRecuperado]) -> str:
        """Genera el texto estable que recibirá el modelo."""
        if not fragmentos:
            return (
                "Sin resultados para esa consulta con esos filtros. "
                "Prueba a quitar algún filtro o a reformular la búsqueda."
            )

        partes = []
        for fragmento in fragmentos:
            puntuacion = (
                f" (puntuación {fragmento.puntuacion:.4f})"
                if fragmento.puntuacion is not None
                else ""
            )
            partes.append(
                f"[{fragmento.chunk_id}] {fragmento.ticker} "
                f"FY{fragmento.fiscal_year} Item {fragmento.item}"
                f"{puntuacion}\n{fragmento.texto}"
            )
        return "\n\n---\n\n".join(partes)

    @staticmethod
    def _normalizar_entrada(
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> tuple[str, str | None, int | None, str | None, int]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query debe ser texto no vacío.")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k debe ser un entero mayor que cero.")
        if fiscal_year is not None and (
            isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int)
        ):
            raise TypeError("fiscal_year debe ser un entero o None.")

        ticker_normalizado = Retriever._normalizar_filtro("ticker", ticker)
        item_normalizado = Retriever._normalizar_filtro("item", item)
        return (
            query.strip(),
            ticker_normalizado,
            fiscal_year,
            item_normalizado,
            k,
        )

    @staticmethod
    def _normalizar_filtro(nombre: str, valor: str | None) -> str | None:
        if valor is None:
            return None
        if not isinstance(valor, str) or not valor.strip():
            raise ValueError(f"{nombre} debe ser texto no vacío o None.")
        return valor.strip().upper()

    @staticmethod
    def _validar_salida(
        fragmentos: tuple[FragmentoRecuperado, ...],
        *,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> None:
        if len(fragmentos) > k:
            raise ValueError(
                f"El retriever devolvió {len(fragmentos)} fragmentos para k={k}."
            )
        if any(
            not isinstance(fragmento, FragmentoRecuperado)
            for fragmento in fragmentos
        ):
            raise TypeError(
                "El retriever debe devolver objetos FragmentoRecuperado."
            )

        ids = [fragmento.chunk_id for fragmento in fragmentos]
        if len(ids) != len(set(ids)):
            raise ValueError("El retriever devolvió chunk_id duplicados.")

        for fragmento in fragmentos:
            if ticker is not None and fragmento.ticker != ticker:
                raise ValueError(
                    f"{fragmento.chunk_id} no cumple el filtro ticker={ticker}."
                )
            if (
                fiscal_year is not None
                and fragmento.fiscal_year != fiscal_year
            ):
                raise ValueError(
                    f"{fragmento.chunk_id} no cumple el filtro "
                    f"fiscal_year={fiscal_year}."
                )
            if item is not None and fragmento.item != item:
                raise ValueError(
                    f"{fragmento.chunk_id} no cumple el filtro item={item}."
                )
