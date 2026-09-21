"""Prueba de integración: el motor real con un modelo guionizado."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from experimentos.hugo.guardrails import (
    MARCA_CIFRAS,
    ComprobadorCifrasXBRL,
    ReparadorCitas,
)
from taller_nlp import (
    ConfiguracionAgente,
    FabricaHerramientas,
    MotorLangChain,
)
from tests.support import TEXTO_2024, crear_corpus_temporal


class ModeloGuionizado(BaseChatModel):
    """Devuelve mensajes preparados y guarda lo que recibe."""

    guion: list[AIMessage]
    recibidos: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "guionizado"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModeloGuionizado":
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self.recibidos.append(list(messages))
        indice = min(len(self.recibidos) - 1, len(self.guion) - 1)
        return ChatResult(
            generations=[ChatGeneration(message=self.guion[indice])]
        )


def _respuesta_estructurada(identificador: str, **campos: Any) -> AIMessage:
    argumentos = {"respuesta": "Respuesta.", "fuente": "xbrl"}
    argumentos.update(campos)
    return AIMessage(
        content="",
        id=f"ai-{identificador}",
        tool_calls=[
            {
                "name": "RespuestaFinanciera",
                "args": argumentos,
                "id": f"tool-{identificador}",
            }
        ],
    )


def _crear_motor(corpus, fragmento, middlewares, guion):
    def buscar(corpus_, query, ticker, fiscal_year, item, k):
        del corpus_, query, ticker, fiscal_year, item, k
        return (
            f"[{fragmento.chunk_id}] ACME FY2024 Item 1A\n{fragmento.texto}"
        )

    fabrica = FabricaHerramientas(corpus, search_filings_impl=buscar)
    configuracion = ConfiguracionAgente(
        modelo="modelo-guionizado",
        system_prompt="Responde con evidencia.",
        max_iteraciones=6,
    )
    modelo = ModeloGuionizado(guion=guion, recibidos=[])
    motor = MotorLangChain(
        configuracion,
        fabrica.crear(),
        corpus,
        middlewares=middlewares,
        modelo=modelo,
    )
    return motor, modelo


class TestGuardrailesEnElMotor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.corpus, self.fragmentos = crear_corpus_temporal(
            Path(self.temporal.name)
        )
        self.fragmento = self.fragmentos[0]

    def tearDown(self) -> None:
        self.temporal.cleanup()

    def test_el_comprobador_devuelve_el_desajuste_y_el_modelo_corrige(self) -> None:
        guion = [
            _respuesta_estructurada(
                "1", cifra=250.0, unidad="USD", ticker="ACME", ejercicio=2024
            ),
            AIMessage(
                content="",
                id="ai-2",
                tool_calls=[
                    {
                        "name": "get_xbrl_fact",
                        "args": {
                            "ticker": "ACME",
                            "fiscal_year": 2024,
                            "concept": "NetIncomeLoss",
                        },
                        "id": "tool-xbrl",
                    }
                ],
            ),
            _respuesta_estructurada(
                "3", cifra=100.0, unidad="USD", ticker="ACME", ejercicio=2024
            ),
        ]
        comprobador = ComprobadorCifrasXBRL(self.corpus)
        motor, modelo = _crear_motor(
            self.corpus,
            self.fragmento,
            (comprobador,),
            guion,
        )

        respuesta = motor.responder("¿Cuál fue el beneficio de ACME en 2024?")

        self.assertEqual(respuesta.cifra, 100.0)
        self.assertIsNone(respuesta.error)
        self.assertEqual(
            [llamada.nombre for llamada in respuesta.llamadas],
            ["get_xbrl_fact"],
        )
        recibido = "".join(
            str(mensaje.content) for mensaje in modelo.recibidos[1]
        )
        self.assertIn(MARCA_CIFRAS, recibido)
        self.assertEqual(
            [i["tipo"] for i in comprobador.intervenciones], ["correccion"]
        )
        self.assertEqual(
            comprobador.intervenciones[0]["pregunta"],
            "¿Cuál fue el beneficio de ACME en 2024?",
        )

    def test_el_reparador_deja_la_cita_literal_en_la_respuesta_final(self) -> None:
        cita_del_modelo = f"«{TEXTO_2024.upper()}»"
        guion = [
            AIMessage(
                content="",
                id="ai-1",
                tool_calls=[
                    {
                        "name": "search_filings",
                        "args": {"query": "supply chain"},
                        "id": "tool-busqueda",
                    }
                ],
            ),
            _respuesta_estructurada(
                "2",
                fuente="texto",
                cita=cita_del_modelo,
                chunk_id=self.fragmento.chunk_id,
            ),
        ]
        motor, _ = _crear_motor(
            self.corpus,
            self.fragmento,
            (ComprobadorCifrasXBRL(self.corpus), ReparadorCitas(self.corpus)),
            guion,
        )

        respuesta = motor.responder("¿Qué riesgos menciona ACME?")

        self.assertEqual(respuesta.cita, TEXTO_2024)
        self.assertEqual(respuesta.citas, (self.fragmento.chunk_id,))


if __name__ == "__main__":
    unittest.main()
