from __future__ import annotations

import unittest

from experimentos.higinio.agente_v009 import crear_constructor as crear_v009
from experimentos.higinio.agente_v011 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v011,
)
from experimentos.higinio.contrato_salida import RespuestaFinancieraTolerante
from experimentos.higinio.guardrail_comparativas import GuardrailComparativas
from experimentos.higinio.verificador_citas import VerificadorCitas
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RespuestaFinanciera


class TestRespuestaFinancieraTolerante(unittest.TestCase):
    @staticmethod
    def _respuesta(**cambios: object) -> RespuestaFinancieraTolerante:
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
        return RespuestaFinancieraTolerante.model_validate(datos)

    def test_normaliza_una_comparativa_completa(self) -> None:
        respuesta = self._respuesta()

        self.assertEqual(respuesta.cifra, 125.0)
        self.assertEqual(respuesta.ejercicio, 2024)
        self.assertEqual(respuesta.variacion_absoluta, 25.0)
        self.assertEqual(respuesta.variacion_porcentual, 25.0)

    def test_acepta_comparativa_incompleta_sin_perder_la_respuesta(self) -> None:
        respuesta = self._respuesta(
            cifra_final=None,
            fuente="xbrl",
        )

        self.assertEqual(respuesta.cifra, 25.0)
        self.assertEqual(respuesta.ejercicio, 2023)
        self.assertIsNone(respuesta.cifra_final)

    def test_acepta_periodos_invertidos_y_normaliza_el_final_declarado(self) -> None:
        respuesta = self._respuesta(
            ejercicio_inicial=2025,
            ejercicio_final=2024,
        )

        self.assertEqual(respuesta.cifra, 125.0)
        self.assertEqual(respuesta.ejercicio, 2024)

    def test_infiere_comparativa_completa_sin_clasificacion(self) -> None:
        respuesta = self._respuesta(tipo_respuesta=None)

        self.assertEqual(respuesta.cifra, 125.0)
        self.assertEqual(respuesta.ejercicio, 2024)
        self.assertEqual(respuesta.variacion_absoluta, 25.0)

    def test_no_altera_una_respuesta_numerica(self) -> None:
        respuesta = RespuestaFinancieraTolerante(
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

    def test_esquema_no_exige_campos_comparativos(self) -> None:
        requeridos = set(
            RespuestaFinancieraTolerante.model_json_schema().get(
                "required", ()
            )
        )

        self.assertEqual(requeridos, {"respuesta", "fuente"})


class TestHiginioV011(unittest.TestCase):
    def test_parte_de_v009_y_solo_sustituye_el_contrato_de_salida(self) -> None:
        v009 = crear_v009()
        v011 = crear_v011()
        retriever_v009 = v009.fabrica_herramientas.retriever
        retriever_v011 = v011.fabrica_herramientas.retriever

        self.assertEqual(v011.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v011.configuracion, v009.configuracion)
        self.assertEqual(v011.corpus, v009.corpus)
        self.assertEqual(v011.evaluador.k_retrieval, v009.evaluador.k_retrieval)
        self.assertIs(v009.esquema_respuesta, RespuestaFinanciera)
        self.assertIs(v011.esquema_respuesta, RespuestaFinancieraTolerante)
        self.assertEqual(len(v011.middlewares), 3)
        self.assertEqual(
            tuple(type(middleware) for middleware in v011.middlewares),
            tuple(type(middleware) for middleware in v009.middlewares),
        )
        self.assertIsInstance(v011.middlewares[0], VerificadorCifrasXBRL)
        self.assertIsInstance(v011.middlewares[1], GuardrailComparativas)
        self.assertIsInstance(v011.middlewares[2], VerificadorCitas)
        self.assertIsNotNone(retriever_v009)
        self.assertIsNotNone(retriever_v011)
        assert retriever_v009 is not None
        assert retriever_v011 is not None
        self.assertEqual(retriever_v011.nombre, retriever_v009.nombre)
        self.assertEqual(
            dict(retriever_v011.parametros),
            dict(retriever_v009.parametros),
        )


if __name__ == "__main__":
    unittest.main()
