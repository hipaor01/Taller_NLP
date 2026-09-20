from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from experimentos.higinio.agente_v001 import crear_constructor as crear_v001
from experimentos.higinio.agente_v005 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v005,
)
from experimentos.higinio.verificador_xbrl import (
    MARCA_VERIFICACION,
    VerificadorCifrasXBRL,
)
from taller_nlp import RespuestaFinanciera
from tests.support import crear_corpus_temporal


class TestVerificadorCifrasXBRL(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        corpus, _ = crear_corpus_temporal(Path(self.temporal.name))
        self.verificador = VerificadorCifrasXBRL(corpus)
        self.runtime = Runtime()

    def tearDown(self) -> None:
        self.temporal.cleanup()

    @staticmethod
    def _respuesta(
        cifra: float | None,
        *,
        ticker: str | None = "ACME",
        ejercicio: int | None = 2024,
    ) -> RespuestaFinanciera:
        return RespuestaFinanciera(
            respuesta="Respuesta de prueba.",
            cifra=cifra,
            ticker=ticker,
            ejercicio=ejercicio,
            fuente="xbrl",
        )

    def test_no_interviene_sin_cifra_identidad_o_si_la_cifra_cuadra(self) -> None:
        casos = (
            self._respuesta(None),
            self._respuesta(100, ticker=None),
            self._respuesta(100, ejercicio=None),
            self._respuesta(100.5),
        )
        for respuesta in casos:
            with self.subTest(respuesta=respuesta):
                self.assertIsNone(
                    self.verificador.after_model(
                        {"messages": [], "structured_response": respuesta},
                        self.runtime,
                    )
                )

    def test_devuelve_mensaje_con_hechos_y_salto_al_modelo(self) -> None:
        resultado = self.verificador.after_model(
            {
                "messages": [],
                "structured_response": self._respuesta(250),
            },
            self.runtime,
        )

        assert resultado is not None
        self.assertEqual(resultado["jump_to"], "model")
        mensaje = resultado["messages"][0]
        self.assertEqual(mensaje["role"], "user")
        self.assertIn(MARCA_VERIFICACION, mensaje["content"])
        self.assertIn("250", mensaje["content"])
        self.assertIn("NetIncomeLoss=100", mensaje["content"])
        self.assertNotIn("NetIncomeLoss=80", mensaje["content"])
        self.assertIn("get_xbrl_fact", mensaje["content"])

    def test_la_marca_impide_una_segunda_correccion(self) -> None:
        resultado = self.verificador.after_model(
            {
                "messages": [
                    HumanMessage(
                        content=f"{MARCA_VERIFICACION}: corrige la cifra"
                    )
                ],
                "structured_response": self._respuesta(250),
            },
            self.runtime,
        )

        self.assertIsNone(resultado)

    def test_declara_el_salto_condicional_al_modelo(self) -> None:
        self.assertEqual(
            getattr(VerificadorCifrasXBRL.after_model, "__can_jump_to__"),
            ["model"],
        )


class TestHiginioV005(unittest.TestCase):
    def test_parte_de_v001_y_solo_anade_el_verificador(self) -> None:
        v001 = crear_v001()
        v005 = crear_v005()
        retriever_v001 = v001.fabrica_herramientas.retriever
        retriever_v005 = v005.fabrica_herramientas.retriever

        self.assertEqual(v005.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v005.configuracion, v001.configuracion)
        self.assertEqual(v005.corpus, v001.corpus)
        self.assertEqual(v005.evaluador.k_retrieval, v001.evaluador.k_retrieval)
        self.assertIsNotNone(retriever_v001)
        self.assertIsNotNone(retriever_v005)
        assert retriever_v001 is not None
        assert retriever_v005 is not None
        self.assertEqual(retriever_v005.nombre, retriever_v001.nombre)
        self.assertEqual(
            dict(retriever_v005.parametros),
            dict(retriever_v001.parametros),
        )
        self.assertEqual(len(v005.middlewares), 1)
        self.assertIsInstance(v005.middlewares[0], VerificadorCifrasXBRL)


if __name__ == "__main__":
    unittest.main()
