"""Factoría que liga implementaciones de tools a una variante del corpus."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TypeAlias

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool

from .contracts import NombreHerramienta
from .corpus import CorpusVariant
from .retrieval import Retriever
from .tool_suite import ToolSuite


ListAvailableImpl: TypeAlias = Callable[[CorpusVariant], str]
GetXbrlFactImpl: TypeAlias = Callable[[CorpusVariant, str, int, str], str]
SearchFilingsImpl: TypeAlias = Callable[
    [CorpusVariant, str, str | None, int | None, str | None, int], str
]
ReadSectionImpl: TypeAlias = Callable[[CorpusVariant, str, int, str], str]


_DESCRIPCIONES: dict[NombreHerramienta, str] = {
    "list_available": """Lista qué compañías, ejercicios y secciones existen
en el corpus.

Úsala SIEMPRE antes de responder que un dato no existe, y antes de llamar a
cualquier otra herramienta si no estás seguro de que la compañía o el ejercicio
que te piden estén en el corpus.""",
    "get_xbrl_fact": """Devuelve el valor EXACTO de una magnitud financiera tal y
como la compañía la reportó en XBRL.

Es la fuente autorizada para cualquier cifra. Úsala SIEMPRE en lugar de leer
un número del texto del informe.

Args:
    ticker: Símbolo bursátil, p. ej. 'NVDA'.
    fiscal_year: Ejercicio fiscal reportado, p. ej. 2024.
    concept: Concepto en taxonomía US-GAAP, p. ej. 'Revenues',
        'NetIncomeLoss', 'Assets', 'OperatingIncomeLoss'.

Devuelve el valor con su unidad y fecha de cierre, o un aviso explícito si la
compañía no reportó ese concepto en ese ejercicio.""",
    "search_filings": """Busca fragmentos de texto relevantes en los informes
10-K del corpus.

Úsala para preguntas cualitativas: riesgos, estrategia, litigios, comentarios
de la dirección. NO la uses para obtener cifras: para eso está get_xbrl_fact.

Args:
    query: Qué buscar, en lenguaje natural.
    ticker: Filtra por compañía si la pregunta la menciona.
    fiscal_year: Filtra por ejercicio si la pregunta lo menciona.
    item: Filtra por sección: '1A' riesgos, '7' MD&A, '7A' riesgo de mercado,
        '8' estados financieros.
    k: Número de fragmentos a devolver.

Devuelve k fragmentos, cada uno con su chunk_id para poder citarlo.""",
    "read_section": """Devuelve el TEXTO COMPLETO de una sección de un 10-K.

Es una herramienta CARA: puede devolver decenas de miles de tokens. Úsala solo
cuando search_filings devuelva fragmentos insuficientes y necesites el contexto
entero de una sección concreta.

Args:
    ticker: Símbolo bursátil, p. ej. 'META'.
    fiscal_year: Ejercicio fiscal, p. ej. 2025.
    item: '1A' riesgos, '7' MD&A, '7A' riesgo de mercado,
        '8' estados financieros.""",
}


class FabricaHerramientas:
    """Construye un ``ToolSuite`` ligado a un único ``CorpusVariant``."""

    def __init__(
        self,
        corpus: CorpusVariant,
        search_filings_impl: SearchFilingsImpl | None = None,
        *,
        retriever: Retriever | None = None,
        list_available_impl: ListAvailableImpl | None = None,
        get_xbrl_fact_impl: GetXbrlFactImpl | None = None,
        read_section_impl: ReadSectionImpl | None = None,
        descripciones: Mapping[NombreHerramienta, str] | None = None,
    ) -> None:
        if (search_filings_impl is None) == (retriever is None):
            raise ValueError(
                "Debe indicarse exactamente uno entre retriever y "
                "search_filings_impl."
            )
        if retriever is not None:
            if not isinstance(retriever, Retriever):
                raise TypeError("retriever debe ser una instancia de Retriever.")
            if retriever.corpus != corpus:
                raise ValueError(
                    "El retriever y la fábrica deben usar la misma "
                    "CorpusVariant."
                )
        if search_filings_impl is not None:
            self._validar_implementacion(
                "search_filings", search_filings_impl, numero_argumentos=6
            )
        implementaciones_opcionales = {
            "list_available": list_available_impl,
            "get_xbrl_fact": get_xbrl_fact_impl,
            "read_section": read_section_impl,
        }
        aridades = {
            "list_available": 1,
            "get_xbrl_fact": 4,
            "read_section": 4,
        }
        for nombre, implementacion in implementaciones_opcionales.items():
            if implementacion is not None:
                self._validar_implementacion(
                    nombre,
                    implementacion,
                    numero_argumentos=aridades[nombre],
                )

        descripciones_recibidas = dict(descripciones or {})
        desconocidas = descripciones_recibidas.keys() - _DESCRIPCIONES.keys()
        if desconocidas:
            raise ValueError(
                f"Descripciones de herramientas desconocidas: "
                f"{sorted(desconocidas)}."
            )
        for nombre, descripcion in descripciones_recibidas.items():
            if not isinstance(descripcion, str) or not descripcion.strip():
                raise ValueError(f"La descripción de {nombre} no puede estar vacía.")

        self._corpus = corpus
        self._retriever = retriever
        self._search_filings_impl = (
            search_filings_impl
            if search_filings_impl is not None
            else self._search_filings_con_retriever
        )
        self._list_available_impl = (
            list_available_impl or self._list_available_baseline
        )
        self._get_xbrl_fact_impl = (
            get_xbrl_fact_impl or self._get_xbrl_fact_baseline
        )
        self._read_section_impl = read_section_impl or self._read_section_baseline
        self._descripciones = {**_DESCRIPCIONES, **descripciones_recibidas}
        if (
            retriever is not None
            and not retriever.aplica_filtros_metadatos
            and "search_filings" not in descripciones_recibidas
        ):
            self._descripciones["search_filings"] += (
                "\n\nNota de esta configuración: ticker, fiscal_year e item "
                "se aceptan para conservar el contrato, pero no restringen "
                "el ranking denso."
            )

        necesita_secciones = list_available_impl is None or read_section_impl is None
        self._secciones = (
            pd.read_json(corpus.ruta_secciones, lines=True)
            if necesita_secciones
            else None
        )
        self._xbrl = (
            pd.read_parquet(corpus.ruta_xbrl)
            if get_xbrl_fact_impl is None
            else None
        )

    @property
    def corpus(self) -> CorpusVariant:
        return self._corpus

    @property
    def retriever(self) -> Retriever | None:
        return self._retriever

    @property
    def implementaciones(self) -> Mapping[NombreHerramienta, Callable]:
        """Callables efectivos usados por cada herramienta."""
        return MappingProxyType(
            {
                "list_available": self._list_available_impl,
                "get_xbrl_fact": self._get_xbrl_fact_impl,
                "search_filings": self._search_filings_impl,
                "read_section": self._read_section_impl,
            }
        )

    @property
    def descripciones(self) -> Mapping[NombreHerramienta, str]:
        """Descripciones efectivas que se entregarán al modelo."""
        return MappingProxyType(dict(self._descripciones))

    def crear(self) -> ToolSuite:
        """Crea las cuatro tools LangChain con sus contratos canónicos."""

        def list_available() -> str:
            return self._list_available_impl(self._corpus)

        def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
            return self._get_xbrl_fact_impl(
                self._corpus, ticker, fiscal_year, concept
            )

        def search_filings(
            query: str,
            ticker: str | None = None,
            fiscal_year: int | None = None,
            item: str | None = None,
            k: int = 5,
        ) -> str:
            return self._search_filings_impl(
                self._corpus, query, ticker, fiscal_year, item, k
            )

        def read_section(ticker: str, fiscal_year: int, item: str) -> str:
            return self._read_section_impl(
                self._corpus, ticker, fiscal_year, item
            )

        return ToolSuite(
            list_available=self._crear_tool("list_available", list_available),
            get_xbrl_fact=self._crear_tool("get_xbrl_fact", get_xbrl_fact),
            search_filings=self._crear_tool("search_filings", search_filings),
            read_section=self._crear_tool("read_section", read_section),
        )

    def _crear_tool(self, nombre: NombreHerramienta, funcion: Callable) -> BaseTool:
        return StructuredTool.from_function(
            func=funcion,
            name=nombre,
            description=self._descripciones[nombre],
        )

    @staticmethod
    def _validar_implementacion(
        nombre: str,
        implementacion: Callable,
        *,
        numero_argumentos: int,
    ) -> None:
        if not callable(implementacion):
            raise TypeError(f"La implementación de {nombre} debe ser callable.")
        try:
            inspect.signature(implementacion).bind(
                *(object() for _ in range(numero_argumentos))
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"La implementación de {nombre} no acepta los "
                f"{numero_argumentos} argumentos posicionales de su contrato."
            ) from exc

    def _list_available_baseline(self, corpus: CorpusVariant) -> str:
        del corpus
        assert self._secciones is not None
        columnas_identidad = ["cik", "ticker", "empresa"]
        if self._secciones[columnas_identidad].isna().any().any():
            raise ValueError("Hay secciones sin CIK, ticker o empresa.")

        por_cik = self._secciones.groupby("cik", sort=True)
        cardinalidad = por_cik[["ticker", "empresa"]].nunique()
        inconsistentes = cardinalidad.index[(cardinalidad != 1).any(axis=1)]
        if len(inconsistentes):
            raise ValueError(
                f"CIK con ticker o empresa no únicos: {inconsistentes.tolist()}"
            )

        lineas = []
        for cik, grupo in por_cik:
            ticker = grupo["ticker"].iloc[0]
            empresa = grupo["empresa"].iloc[0]
            ejercicios = []
            for fiscal_year, grupo_year in grupo.groupby(
                "fiscal_year", sort=True
            ):
                items = sorted(grupo_year["item"].unique())
                ejercicios.append(f"FY{int(fiscal_year)} [{', '.join(items)}]")
            lineas.append(
                f"- **{ticker}** — {empresa} (CIK {cik}): "
                f"{'; '.join(ejercicios)}"
            )
        return "\n".join(lineas)

    def _get_xbrl_fact_baseline(
        self,
        corpus: CorpusVariant,
        ticker: str,
        fiscal_year: int,
        concept: str,
    ) -> str:
        del corpus
        assert self._xbrl is not None
        filas = self._xbrl[
            (self._xbrl.ticker == ticker)
            & (self._xbrl.fiscal_year.astype(int) == int(fiscal_year))
            & (self._xbrl.concept == concept)
        ]
        if filas.empty:
            disponibles = sorted(
                self._xbrl[
                    (self._xbrl.ticker == ticker)
                    & (self._xbrl.fiscal_year.astype(int) == int(fiscal_year))
                ].concept.unique()
            )
            if not disponibles:
                return (
                    f"No hay datos de {ticker} para FY{fiscal_year} en el "
                    "corpus. Usa list_available para ver qué hay."
                )
            return (
                f"{ticker} no reportó {concept!r} en FY{fiscal_year}. "
                f"Conceptos disponibles: {', '.join(disponibles)}"
            )
        fila = filas.iloc[0]
        return (
            f"{ticker} FY{fiscal_year} · {concept} = {fila.value:,.0f} "
            f"{fila.unit} (cierre de ejercicio {fila.period_end}, "
            f"según el {fila.form})"
        )

    def _read_section_baseline(
        self,
        corpus: CorpusVariant,
        ticker: str,
        fiscal_year: int,
        item: str,
    ) -> str:
        del corpus
        assert self._secciones is not None
        filas = self._secciones[
            (self._secciones.ticker == ticker)
            & (self._secciones.fiscal_year.astype(int) == int(fiscal_year))
            & (self._secciones.item == item)
        ]
        if filas.empty:
            return (
                f"No hay Item {item} de {ticker} FY{fiscal_year} en el "
                "corpus. Usa list_available para ver qué hay."
            )
        return str(filas.iloc[0].texto)

    def _search_filings_con_retriever(
        self,
        corpus: CorpusVariant,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> str:
        del corpus
        assert self._retriever is not None
        return self._retriever.buscar_formateado(
            query=query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            item=item,
            k=k,
        )
