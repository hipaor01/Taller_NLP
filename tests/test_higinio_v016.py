from __future__ import annotations

import unittest

from experimentos.higinio.agente_v015 import crear_constructor as crear_v015
from experimentos.higinio.agente_v016 import (
    MODELO,
    NOMBRE_VARIANTE,
    PRECIO_ENTRADA_USD_MILLON_TOKENS,
    PRECIO_SALIDA_USD_MILLON_TOKENS,
    crear_constructor as crear_v016,
)


class TestHiginioV016(unittest.TestCase):
    def test_parte_de_v015_y_solo_sustituye_modelo_y_tarifas(self) -> None:
        v015 = crear_v015()
        v016 = crear_v016()

        self.assertEqual(v016.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v016.configuracion.modelo, MODELO)
        self.assertEqual(
            v016.configuracion.precio_entrada_usd_millon_tokens,
            PRECIO_ENTRADA_USD_MILLON_TOKENS,
        )
        self.assertEqual(
            v016.configuracion.precio_salida_usd_millon_tokens,
            PRECIO_SALIDA_USD_MILLON_TOKENS,
        )
        self.assertEqual(
            v016.configuracion.model_dump(
                exclude={
                    "modelo",
                    "precio_entrada_usd_millon_tokens",
                    "precio_salida_usd_millon_tokens",
                }
            ),
            v015.configuracion.model_dump(
                exclude={
                    "modelo",
                    "precio_entrada_usd_millon_tokens",
                    "precio_salida_usd_millon_tokens",
                }
            ),
        )
        self.assertEqual(v016.corpus, v015.corpus)
        retriever_v015 = v015.fabrica_herramientas.retriever
        retriever_v016 = v016.fabrica_herramientas.retriever
        self.assertIsNotNone(retriever_v015)
        self.assertIsNotNone(retriever_v016)
        assert retriever_v015 is not None
        assert retriever_v016 is not None
        self.assertIs(type(retriever_v016), type(retriever_v015))
        self.assertEqual(retriever_v016.nombre, retriever_v015.nombre)
        self.assertEqual(
            dict(retriever_v016.parametros),
            dict(retriever_v015.parametros),
        )
        self.assertIs(v016.esquema_respuesta, v015.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v016.middlewares),
            tuple(type(middleware) for middleware in v015.middlewares),
        )
        self.assertIs(v016.modelo, v015.modelo)
        self.assertEqual(
            dict(v016.control_peticiones.parametros),
            dict(v015.control_peticiones.parametros),
        )
        self.assertIs(v016.telemetria_auxiliar, v015.telemetria_auxiliar)
        self.assertEqual(
            v016.evaluador.k_retrieval,
            v015.evaluador.k_retrieval,
        )


if __name__ == "__main__":
    unittest.main()
