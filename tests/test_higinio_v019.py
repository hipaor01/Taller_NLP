from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from experimentos.higinio.agente_v018 import crear_constructor as crear_v018
from experimentos.higinio.agente_v019 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v019,
)
from experimentos.higinio.recuperador_salida import (
    RecuperadorSalidaEstructurada,
)
from taller_nlp import RespuestaFinanciera
from tests.support import TEXTO_2024, crear_corpus_temporal


class TestRecuperadorSalidaEstructurada(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.corpus, fragmentos = crear_corpus_temporal(
            Path(self.temporal.name)
        )
        self.fragmento = fragmentos[0]
        self.recuperador = RecuperadorSalidaEstructurada(
            self.corpus,
            RespuestaFinanciera,
        )
        self.runtime = Runtime()

    def _mensajes(
        self,
        texto_final: str,
        *,
        nombre_herramienta: str = "search_filings",
        consulta: str = "supply chain disruptions materially harm operations",
    ) -> list[object]:
        id_llamada = "tool-1"
        return [
            HumanMessage(content="¿Qué riesgo describe la compañía?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": nombre_herramienta,
                        "args": {"query": consulta},
                        "id": id_llamada,
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=(
                    f"[{self.fragmento.chunk_id}] ACME FY2024 Item 1A\n"
                    f"{TEXTO_2024}"
                ),
                name=nombre_herramienta,
                tool_call_id=id_llamada,
            ),
            AIMessage(
                content=texto_final,
            ),
        ]

    def test_recupera_texto_final_con_evidencia_literal_univoca(self) -> None:
        resultado = self.recuperador.after_model(
            {
                "messages": self._mensajes(
                    "La compañía advierte de interrupciones operativas."
                ),
                "structured_response": None,
            },
            self.runtime,
        )

        assert resultado is not None
        respuesta = resultado["structured_response"]
        self.assertEqual(
            respuesta.respuesta,
            "La compañía advierte de interrupciones operativas.",
        )
        self.assertEqual(respuesta.fuente, "texto")
        self.assertEqual(respuesta.cita, TEXTO_2024)
        self.assertEqual(respuesta.chunk_id, self.fragmento.chunk_id)
        self.assertEqual(respuesta.ticker, "ACME")
        self.assertEqual(respuesta.ejercicio, 2024)

    def test_recupera_json_textual_solo_si_la_cita_es_verificable(self) -> None:
        contenido = json.dumps(
            {
                "respuesta": "La cadena de suministro puede afectar operaciones.",
                "cifra": None,
                "unidad": None,
                "ticker": "ACME",
                "ejercicio": 2024,
                "fuente": "texto",
                "cita": TEXTO_2024,
                "chunk_id": self.fragmento.chunk_id,
            }
        )

        resultado = self.recuperador.after_model(
            {
                "messages": self._mensajes(contenido),
                "structured_response": None,
            },
            self.runtime,
        )

        assert resultado is not None
        respuesta = resultado["structured_response"]
        self.assertEqual(
            respuesta.respuesta,
            "La cadena de suministro puede afectar operaciones.",
        )
        self.assertEqual(respuesta.cita, TEXTO_2024)

    def test_recupera_json_de_una_llamada_estructurada_invalida(self) -> None:
        contenido = json.dumps(
            {
                "respuesta": "La cadena de suministro puede afectar operaciones.",
                "cifra": None,
                "unidad": None,
                "ticker": "ACME",
                "ejercicio": 2024,
                "fuente": "texto",
                "cita": TEXTO_2024,
                "chunk_id": self.fragmento.chunk_id,
            }
        )
        mensajes = self._mensajes("")
        mensajes[-1] = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "RespuestaFinanciera",
                    "args": contenido,
                    "id": "respuesta-1",
                    "error": "La herramienta no fue reconocida.",
                    "type": "invalid_tool_call",
                }
            ],
        )

        resultado = self.recuperador.after_model(
            {
                "messages": mensajes,
                "structured_response": None,
            },
            self.runtime,
        )

        assert resultado is not None
        respuesta = resultado["structured_response"]
        self.assertEqual(respuesta.fuente, "texto")
        self.assertEqual(respuesta.chunk_id, self.fragmento.chunk_id)

    def test_no_interviene_si_ya_hay_respuesta_estructurada(self) -> None:
        estructurada = RespuestaFinanciera(
            respuesta="Respuesta existente.",
            fuente="texto",
            cita=TEXTO_2024,
            chunk_id=self.fragmento.chunk_id,
        )

        resultado = self.recuperador.after_model(
            {
                "messages": self._mensajes("Texto alternativo."),
                "structured_response": estructurada,
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_no_interviene_en_una_traza_xbrl(self) -> None:
        resultado = self.recuperador.after_model(
            {
                "messages": self._mensajes(
                    "El beneficio fue de 100 USD.",
                    nombre_herramienta="get_xbrl_fact",
                    consulta="NetIncomeLoss",
                ),
                "structured_response": None,
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_no_interviene_sin_evidencia_inequivoca(self) -> None:
        resultado = self.recuperador.after_model(
            {
                "messages": self._mensajes(
                    "Respuesta sin relación literal.",
                    consulta="unrelated topic",
                ),
                "structured_response": None,
            },
            self.runtime,
        )

        self.assertIsNone(resultado)


class TestHiginioV019(unittest.TestCase):
    def test_parte_de_v018_y_solo_anade_el_fallback(self) -> None:
        v018 = crear_v018()
        v019 = crear_v019()

        self.assertEqual(v019.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v019.configuracion, v018.configuracion)
        self.assertEqual(v019.corpus, v018.corpus)
        self.assertIs(v019.esquema_respuesta, v018.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v019.middlewares[:-1]),
            tuple(type(middleware) for middleware in v018.middlewares),
        )
        self.assertIsInstance(
            v019.middlewares[-1],
            RecuperadorSalidaEstructurada,
        )


if __name__ == "__main__":
    unittest.main()
