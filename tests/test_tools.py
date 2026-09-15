from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.tools import StructuredTool

from taller_nlp import FabricaHerramientas, ToolSuite

from tests.support import (
    RetrieverControlado,
    como_recuperado,
    crear_corpus_temporal,
)


def crear_tool(nombre: str, funcion):
    return StructuredTool.from_function(
        func=funcion,
        name=nombre,
        description=f"Tool {nombre}",
    )


def suite_valida() -> ToolSuite:
    def list_available() -> str:
        return "ok"

    def get_xbrl_fact(ticker: str, fiscal_year: int, concept: str) -> str:
        return f"{ticker}-{fiscal_year}-{concept}"

    def search_filings(
        query: str,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        item: str | None = None,
        k: int = 5,
    ) -> str:
        return query

    def read_section(ticker: str, fiscal_year: int, item: str) -> str:
        return f"{ticker}-{fiscal_year}-{item}"

    return ToolSuite(
        list_available=crear_tool("list_available", list_available),
        get_xbrl_fact=crear_tool("get_xbrl_fact", get_xbrl_fact),
        search_filings=crear_tool("search_filings", search_filings),
        read_section=crear_tool("read_section", read_section),
    )


class TestToolSuite(unittest.TestCase):
    def test_expone_orden_y_nombres_canonicos(self) -> None:
        suite = suite_valida()
        self.assertEqual(
            [tool.name for tool in suite.herramientas],
            ["list_available", "get_xbrl_fact", "search_filings", "read_section"],
        )
        self.assertEqual(set(suite.por_nombre), set(tool.name for tool in suite.herramientas))

    def test_rechaza_nombre_o_esquema_incorrectos(self) -> None:
        suite = suite_valida()

        def sin_argumentos() -> str:
            return "x"

        with self.assertRaisesRegex(ValueError, "debe conservar el nombre"):
            suite.reemplazar(
                list_available=crear_tool("otro_nombre", sin_argumentos)
            )
        with self.assertRaisesRegex(ValueError, "no respeta el contrato"):
            suite.reemplazar(
                get_xbrl_fact=crear_tool("get_xbrl_fact", sin_argumentos)
            )

    def test_reemplazar_no_muta_la_suite_original(self) -> None:
        suite = suite_valida()

        def list_available() -> str:
            return "nuevo"

        nueva_tool = crear_tool("list_available", list_available)
        nueva = suite.reemplazar(list_available=nueva_tool)
        self.assertIsNot(suite, nueva)
        self.assertIs(nueva.list_available, nueva_tool)
        self.assertIsNot(suite.list_available, nueva_tool)


class TestFabricaHerramientas(unittest.TestCase):
    def test_liga_retriever_y_genera_las_cuatro_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            retriever = RetrieverControlado(
                corpus, (como_recuperado(fragmentos[0]),)
            )
            suite = FabricaHerramientas(
                corpus, retriever=retriever
            ).crear()
            salida = suite.search_filings.invoke(
                {"query": "risk", "ticker": "ACME", "k": 1}
            )
            self.assertIn("ACME-2024-1A-0000", salida)
            self.assertIs(retriever.corpus, corpus)

    def test_baseline_lista_lee_seccion_y_extrae_xbrl(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))

            def buscar(corpus, query, ticker, fiscal_year, item, k):
                del corpus, query, ticker, fiscal_year, item, k
                return "resultado"

            suite = FabricaHerramientas(
                corpus, search_filings_impl=buscar
            ).crear()
            self.assertIn("**ACME** — Acme Corporation", suite.list_available.invoke({}))
            self.assertEqual(
                suite.read_section.invoke(
                    {"ticker": "ACME", "fiscal_year": 2024, "item": "1A"}
                ),
                "Supply chain disruptions could materially harm operations.",
            )
            self.assertIn(
                "NetIncomeLoss = 100 USD",
                suite.get_xbrl_fact.invoke(
                    {
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "concept": "NetIncomeLoss",
                    }
                ),
            )

    def test_exige_exactamente_retriever_o_implementacion_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))
            with self.assertRaisesRegex(ValueError, "exactamente uno"):
                FabricaHerramientas(corpus)


if __name__ == "__main__":
    unittest.main()
