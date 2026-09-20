from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from experimentos.higinio.agente_v005 import crear_constructor as crear_v005
from experimentos.higinio.agente_v008 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v008,
)
from experimentos.higinio.guardrail_comparativas import (
    MARCA_GUARDRAIL_COMPARATIVO,
    GuardrailComparativas,
)
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RespuestaFinanciera


class TestGuardrailComparativas(unittest.TestCase):
    def setUp(self) -> None:
        self.guardrail = GuardrailComparativas()
        self.runtime = Runtime()
        self.pregunta = HumanMessage(
            content=(
                "¿Cómo evolucionó el resultado operativo entre FY2024 "
                "y FY2025?"
            )
        )

    @staticmethod
    def _respuesta(
        *,
        cita: str | None = None,
        chunk_id: str | None = None,
    ) -> RespuestaFinanciera:
        return RespuestaFinanciera(
            respuesta="El resultado operativo aumentó.",
            cifra=147.0,
            unidad="porcentaje",
            ticker="ACME",
            ejercicio=2025,
            fuente="ambas",
            cita=cita,
            chunk_id=chunk_id,
        )

    @staticmethod
    def _busqueda(
        *,
        contenido: str = "[ACME-2025-7-0001] ACME FY2025 Item 7\nEvidence.",
    ) -> list[AIMessage | ToolMessage]:
        return [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_filings",
                        "args": {"query": "operating income change"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=contenido,
                name="search_filings",
                tool_call_id="search-1",
            ),
        ]

    def test_devuelve_comparativa_sin_busqueda_al_modelo(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [self.pregunta],
                "structured_response": self._respuesta(),
            },
            self.runtime,
        )

        assert resultado is not None
        self.assertEqual(resultado["jump_to"], "model")
        mensaje = resultado["messages"][0]["content"]
        self.assertIn(MARCA_GUARDRAIL_COMPARATIVO, mensaje)
        self.assertIn("search_filings", mensaje)
        self.assertIn("get_xbrl_fact", mensaje)

    def test_devuelve_comparativa_sin_cita_recuperada_al_modelo(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [self.pregunta, *self._busqueda()],
                "structured_response": self._respuesta(),
            },
            self.runtime,
        )

        assert resultado is not None
        self.assertEqual(resultado["jump_to"], "model")
        self.assertIn("cita literal", resultado["messages"][0]["content"])

    def test_acepta_comparativa_con_busqueda_y_cita_recuperada(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [self.pregunta, *self._busqueda()],
                "structured_response": self._respuesta(
                    cita="Evidence.",
                    chunk_id="ACME-2025-7-0001",
                ),
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_no_interviene_en_pregunta_no_comparativa(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [
                    HumanMessage(
                        content="¿Cuál fue el resultado operativo en FY2025?"
                    )
                ],
                "structured_response": self._respuesta(),
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_la_marca_impide_una_segunda_correccion(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [
                    self.pregunta,
                    HumanMessage(
                        content=(
                            f"{MARCA_GUARDRAIL_COMPARATIVO}: corrige "
                            "la evidencia."
                        )
                    ),
                ],
                "structured_response": self._respuesta(),
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_busqueda_sin_resultados_no_satisface_el_guardrail(self) -> None:
        resultado = self.guardrail.after_model(
            {
                "messages": [
                    self.pregunta,
                    *self._busqueda(contenido="Sin resultados."),
                ],
                "structured_response": self._respuesta(),
            },
            self.runtime,
        )

        self.assertIsNotNone(resultado)

    def test_declara_el_salto_condicional_al_modelo(self) -> None:
        self.assertEqual(
            getattr(GuardrailComparativas.after_model, "__can_jump_to__"),
            ["model"],
        )


class TestHiginioV008(unittest.TestCase):
    def test_parte_de_v005_y_solo_anade_el_guardrail_comparativo(self) -> None:
        v005 = crear_v005()
        v008 = crear_v008()
        retriever_v005 = v005.fabrica_herramientas.retriever
        retriever_v008 = v008.fabrica_herramientas.retriever

        self.assertEqual(v008.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v008.configuracion, v005.configuracion)
        self.assertEqual(v008.corpus, v005.corpus)
        self.assertEqual(v008.evaluador.k_retrieval, v005.evaluador.k_retrieval)
        self.assertIsNotNone(retriever_v005)
        self.assertIsNotNone(retriever_v008)
        assert retriever_v005 is not None
        assert retriever_v008 is not None
        self.assertEqual(retriever_v008.nombre, retriever_v005.nombre)
        self.assertEqual(
            dict(retriever_v008.parametros),
            dict(retriever_v005.parametros),
        )
        self.assertEqual(len(v008.middlewares), 2)
        self.assertIsInstance(v008.middlewares[0], VerificadorCifrasXBRL)
        self.assertIsInstance(v008.middlewares[1], GuardrailComparativas)


if __name__ == "__main__":
    unittest.main()
