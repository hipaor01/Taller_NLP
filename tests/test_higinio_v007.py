from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from experimentos.higinio.agente_v005 import crear_constructor as crear_v005
from experimentos.higinio.agente_v007 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v007,
)
from experimentos.higinio.verificador_citas import VerificadorCitas
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RespuestaFinanciera
from tests.support import TEXTO_2024, crear_corpus_temporal


class TestVerificadorCitas(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        corpus, fragmentos = crear_corpus_temporal(Path(self.temporal.name))
        self.verificador = VerificadorCitas(corpus)
        self.chunk_id = fragmentos[0].chunk_id
        self.runtime = Runtime()

    def tearDown(self) -> None:
        self.temporal.cleanup()

    def _respuesta(
        self,
        cita: str | None,
        *,
        chunk_id: str | None,
    ) -> RespuestaFinanciera:
        return RespuestaFinanciera(
            respuesta="Respuesta apoyada en el informe.",
            fuente="ambas",
            cita=cita,
            chunk_id=chunk_id,
        )

    def _mensajes_busqueda(self) -> list[AIMessage | ToolMessage]:
        id_llamada = "search-1"
        return [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_filings",
                        "args": {"query": "supply chain"},
                        "id": id_llamada,
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    f"[{self.chunk_id}] ACME FY2024 Item 1A\n{TEXTO_2024}"
                ),
                name="search_filings",
                tool_call_id=id_llamada,
            ),
        ]

    def test_asocia_chunk_si_la_cita_literal_tiene_coincidencia_unica(self) -> None:
        respuesta = self._respuesta(TEXTO_2024, chunk_id=None)

        resultado = self.verificador.after_model(
            {
                "messages": self._mensajes_busqueda(),
                "structured_response": respuesta,
            },
            self.runtime,
        )

        assert resultado is not None
        reparada = resultado["structured_response"]
        self.assertEqual(reparada.chunk_id, self.chunk_id)
        self.assertEqual(reparada.cita, TEXTO_2024)

    def test_literaliza_una_cita_reformateada_en_el_chunk_declarado(self) -> None:
        respuesta = self._respuesta(
            "Supply-chain disruptions could materially harm operations.",
            chunk_id=self.chunk_id,
        )

        resultado = self.verificador.after_model(
            {
                "messages": self._mensajes_busqueda(),
                "structured_response": respuesta,
            },
            self.runtime,
        )

        assert resultado is not None
        reparada = resultado["structured_response"]
        self.assertEqual(reparada.chunk_id, self.chunk_id)
        self.assertEqual(reparada.cita, TEXTO_2024)

    def test_selecciona_linea_tabular_por_etiqueta_y_valores(self) -> None:
        texto = (
            "Jan 26, 2025\t\tJan 28, 2024\t\tChange\n"
            "Revenue\t$\t130,497\t\t\t$\t60,922\n"
            "Operating income\t$\t81,453\t\t\t$\t32,972\t\t\tUp 147%"
        )
        cita_modelo = (
            "Operating income: Jan 26, 2025: $81,453 [million], "
            "Jan 28, 2024: $32,972 [million], Change: Up 147%"
        )

        linea = VerificadorCitas._seleccionar_linea_literal(cita_modelo, texto)

        self.assertEqual(
            linea,
            "Operating income\t$\t81,453\t\t\t$\t32,972\t\t\tUp 147%",
        )

    def test_no_interviene_si_la_cita_ya_es_valida(self) -> None:
        respuesta = self._respuesta(TEXTO_2024, chunk_id=self.chunk_id)

        resultado = self.verificador.after_model(
            {
                "messages": self._mensajes_busqueda(),
                "structured_response": respuesta,
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_no_usa_un_chunk_que_no_fue_recuperado(self) -> None:
        respuesta = self._respuesta(
            "Supply-chain disruptions could materially harm operations.",
            chunk_id=self.chunk_id,
        )

        resultado = self.verificador.after_model(
            {"messages": [], "structured_response": respuesta},
            self.runtime,
        )

        self.assertIsNone(resultado)


class TestHiginioV007(unittest.TestCase):
    def test_parte_de_v005_y_solo_anade_el_verificador_de_citas(self) -> None:
        v005 = crear_v005()
        v007 = crear_v007()
        retriever_v005 = v005.fabrica_herramientas.retriever
        retriever_v007 = v007.fabrica_herramientas.retriever

        self.assertEqual(v007.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v007.configuracion, v005.configuracion)
        self.assertEqual(v007.corpus, v005.corpus)
        self.assertEqual(v007.evaluador.k_retrieval, v005.evaluador.k_retrieval)
        self.assertIsNotNone(retriever_v005)
        self.assertIsNotNone(retriever_v007)
        assert retriever_v005 is not None
        assert retriever_v007 is not None
        self.assertEqual(retriever_v007.nombre, retriever_v005.nombre)
        self.assertEqual(
            dict(retriever_v007.parametros),
            dict(retriever_v005.parametros),
        )
        self.assertEqual(len(v007.middlewares), 2)
        self.assertIsInstance(v007.middlewares[0], VerificadorCifrasXBRL)
        self.assertIsInstance(v007.middlewares[1], VerificadorCitas)


if __name__ == "__main__":
    unittest.main()
