from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from experimentos.higinio.agente_v008 import crear_constructor as crear_v008
from experimentos.higinio.agente_v009 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v009,
)
from experimentos.higinio.guardrail_comparativas import GuardrailComparativas
from experimentos.higinio.verificador_citas import VerificadorCitas
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RespuestaFinanciera
from tests.support import TEXTO_2024, crear_corpus_temporal


class TestComposicionGuardrailYVerificador(unittest.TestCase):
    def test_una_cita_reparada_satisface_el_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            chunk_id = fragmentos[0].chunk_id
            verificador = VerificadorCitas(corpus)
            guardrail = GuardrailComparativas()
            runtime = Runtime()
            id_llamada = "search-1"
            mensajes = [
                HumanMessage(
                    content=(
                        "¿Cómo cambió el beneficio entre FY2023 y FY2024?"
                    )
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_filings",
                            "args": {"query": "profit change"},
                            "id": id_llamada,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=(
                        f"[{chunk_id}] ACME FY2024 Item 1A\n{TEXTO_2024}"
                    ),
                    name="search_filings",
                    tool_call_id=id_llamada,
                ),
            ]
            respuesta = RespuestaFinanciera(
                respuesta="El beneficio aumentó.",
                cifra=25.0,
                unidad="porcentaje",
                ticker="ACME",
                ejercicio=2024,
                fuente="ambas",
                cita=TEXTO_2024,
                chunk_id=None,
            )

            reparacion = verificador.after_model(
                {
                    "messages": mensajes,
                    "structured_response": respuesta,
                },
                runtime,
            )

            assert reparacion is not None
            reparada = reparacion["structured_response"]
            self.assertEqual(reparada.chunk_id, chunk_id)
            self.assertIsNone(
                guardrail.after_model(
                    {
                        "messages": mensajes,
                        "structured_response": reparada,
                    },
                    runtime,
                )
            )


class TestHiginioV009(unittest.TestCase):
    def test_parte_de_v008_y_solo_anade_el_verificador_de_citas(self) -> None:
        v008 = crear_v008()
        v009 = crear_v009()
        retriever_v008 = v008.fabrica_herramientas.retriever
        retriever_v009 = v009.fabrica_herramientas.retriever

        self.assertEqual(v009.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v009.configuracion, v008.configuracion)
        self.assertEqual(v009.corpus, v008.corpus)
        self.assertEqual(v009.evaluador.k_retrieval, v008.evaluador.k_retrieval)
        self.assertIsNotNone(retriever_v008)
        self.assertIsNotNone(retriever_v009)
        assert retriever_v008 is not None
        assert retriever_v009 is not None
        self.assertEqual(retriever_v009.nombre, retriever_v008.nombre)
        self.assertEqual(
            dict(retriever_v009.parametros),
            dict(retriever_v008.parametros),
        )
        self.assertEqual(len(v009.middlewares), 3)
        self.assertIsInstance(v009.middlewares[0], VerificadorCifrasXBRL)
        self.assertIsInstance(v009.middlewares[1], GuardrailComparativas)
        self.assertIsInstance(v009.middlewares[2], VerificadorCitas)


if __name__ == "__main__":
    unittest.main()
