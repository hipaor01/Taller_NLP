from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from taller_nlp import ConfiguracionAgente, MotorLangChain
from taller_nlp import ControlPeticionesModelo


class TestTrazaLangChain(unittest.TestCase):
    def test_materializa_limites_globales_y_por_herramienta(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
            max_iteraciones=3,
            max_llamadas_total=4,
            limites_por_herramienta={
                "read_section": 0,
                "search_filings": 2,
            },
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._middlewares_usuario = ()
        motor._control_peticiones = ControlPeticionesModelo.desde_configuracion(
            configuracion
        )

        middlewares = motor._crear_middlewares()
        limites = [
            (
                type(middleware).__name__,
                getattr(middleware, "tool_name", None),
                getattr(middleware, "run_limit", None),
            )
            for middleware in middlewares
        ]
        self.assertIn(("ModelCallLimitMiddleware", None, 3), limites)
        self.assertIn(("ToolCallLimitMiddleware", None, 4), limites)
        self.assertIn(("ToolCallLimitMiddleware", "read_section", 0), limites)
        self.assertIn(("ToolCallLimitMiddleware", "search_filings", 2), limites)

    def test_extrae_llamadas_chunk_ids_duracion_y_errores(self) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_filings",
                    "args": {"query": "risk"},
                    "id": "call-1",
                    "type": "tool_call",
                },
                {
                    "name": "get_xbrl_fact",
                    "args": {"ticker": "ACME"},
                    "id": "call-2",
                    "type": "tool_call",
                },
            ],
        )
        resultado = ToolMessage(
            content="[ACME-2024-1A-0000] ACME FY2024 Item 1A\ntexto",
            tool_call_id="call-1",
            name="search_filings",
            additional_kwargs={"taller_nlp_duracion_ms": 12.5},
        )
        error = ToolMessage(
            content="fallo de XBRL",
            tool_call_id="call-2",
            name="get_xbrl_fact",
            status="error",
        )
        llamadas = MotorLangChain._extraer_llamadas((ai, resultado, error))
        self.assertEqual(llamadas[0].chunk_ids, ("ACME-2024-1A-0000",))
        self.assertEqual(llamadas[0].duracion_ms, 12.5)
        self.assertTrue(llamadas[0].exitosa)
        self.assertEqual(llamadas[1].error, "fallo de XBRL")
        self.assertFalse(llamadas[1].exitosa)

    def test_registra_tool_call_sin_resultado_como_error(self) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_section",
                    "args": {"ticker": "ACME", "fiscal_year": 2024, "item": "1A"},
                    "id": "call-ausente",
                    "type": "tool_call",
                }
            ],
        )
        llamada = MotorLangChain._extraer_llamadas((ai,))[0]
        self.assertIn("no produjo un ToolMessage", llamada.error or "")

    def test_suma_tokens_y_coste_de_multiples_mensajes(self) -> None:
        mensajes = (
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                response_metadata={"cost": 0.01},
            ),
            AIMessage(
                content="b",
                usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                response_metadata={"cost": 0.02},
            ),
        )
        self.assertEqual(MotorLangChain._extraer_uso(mensajes), (15, 5, 0.03))


if __name__ == "__main__":
    unittest.main()
