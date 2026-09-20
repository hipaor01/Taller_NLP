from __future__ import annotations

import unittest

from pydantic import ValidationError

from experimentos.higinio.agente_v009 import crear_constructor as crear_v009
from experimentos.higinio.agente_v010 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v010,
)
from experimentos.higinio.contrato_salida import RespuestaFinancieraEstricta
from experimentos.higinio.guardrail_comparativas import GuardrailComparativas
from experimentos.higinio.verificador_citas import VerificadorCitas
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RespuestaFinanciera


class TestRespuestaFinancieraEstricta(unittest.TestCase):
    @staticmethod
    def _comparativa(**cambios: object) -> RespuestaFinancieraEstricta:
        datos = {
            "respuesta": "El valor aumentó de 100 a 125 USD.",
            "tipo_respuesta": "comparativa",
            "cifra": 25.0,
            "unidad": "USD",
            "ticker": "ACME",
            "ejercicio": 2023,
            "fuente": "ambas",
            "cita": "The value increased.",
            "chunk_id": "ACME-2024-7-0001",
            "ejercicio_inicial": 2023,
            "cifra_inicial": 100.0,
            "ejercicio_final": 2024,
            "cifra_final": 125.0,
            "variacion_absoluta": -999.0,
            "variacion_porcentual": -999.0,
        }
        datos.update(cambios)
        return RespuestaFinancieraEstricta.model_validate(datos)

    def test_normaliza_cifra_ejercicio_y_variaciones_al_periodo_final(self) -> None:
        respuesta = self._comparativa()

        self.assertEqual(respuesta.cifra, 125.0)
        self.assertEqual(respuesta.ejercicio, 2024)
        self.assertEqual(respuesta.variacion_absoluta, 25.0)
        self.assertEqual(respuesta.variacion_porcentual, 25.0)

    def test_rechaza_comparativa_incompleta(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cifra_final"):
            self._comparativa(cifra_final=None)

    def test_rechaza_periodos_invertidos(self) -> None:
        with self.assertRaisesRegex(ValidationError, "posterior"):
            self._comparativa(ejercicio_inicial=2025, ejercicio_final=2024)

    def test_rechaza_comparativa_que_no_declara_fuente_ambas(self) -> None:
        with self.assertRaisesRegex(ValidationError, "fuente='ambas'"):
            self._comparativa(fuente="xbrl")

    def test_no_altera_una_respuesta_numerica(self) -> None:
        respuesta = RespuestaFinancieraEstricta(
            respuesta="El valor fue 100 USD.",
            tipo_respuesta="numerica",
            cifra=100.0,
            unidad="USD",
            ticker="ACME",
            ejercicio=2024,
            fuente="xbrl",
        )

        self.assertEqual(respuesta.cifra, 100.0)
        self.assertEqual(respuesta.ejercicio, 2024)
        self.assertIsNone(respuesta.cifra_final)


class TestHiginioV010(unittest.TestCase):
    def test_parte_de_v009_y_solo_sustituye_el_contrato_de_salida(self) -> None:
        v009 = crear_v009()
        v010 = crear_v010()
        retriever_v009 = v009.fabrica_herramientas.retriever
        retriever_v010 = v010.fabrica_herramientas.retriever

        self.assertEqual(v010.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v010.configuracion, v009.configuracion)
        self.assertEqual(v010.corpus, v009.corpus)
        self.assertEqual(v010.evaluador.k_retrieval, v009.evaluador.k_retrieval)
        self.assertIs(v009.esquema_respuesta, RespuestaFinanciera)
        self.assertIs(v010.esquema_respuesta, RespuestaFinancieraEstricta)
        self.assertEqual(len(v010.middlewares), 3)
        self.assertEqual(
            tuple(type(middleware) for middleware in v010.middlewares),
            tuple(type(middleware) for middleware in v009.middlewares),
        )
        self.assertIsInstance(v010.middlewares[0], VerificadorCifrasXBRL)
        self.assertIsInstance(v010.middlewares[1], GuardrailComparativas)
        self.assertIsInstance(v010.middlewares[2], VerificadorCitas)
        self.assertIsNotNone(retriever_v009)
        self.assertIsNotNone(retriever_v010)
        assert retriever_v009 is not None
        assert retriever_v010 is not None
        self.assertEqual(retriever_v010.nombre, retriever_v009.nombre)
        self.assertEqual(
            dict(retriever_v010.parametros),
            dict(retriever_v009.parametros),
        )


if __name__ == "__main__":
    unittest.main()
