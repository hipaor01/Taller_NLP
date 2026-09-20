from __future__ import annotations

import unittest

from experimentos.higinio.agente_v002 import KK_RRF
from experimentos.higinio.agente_v012 import crear_constructor as crear_v012
from experimentos.higinio.agente_v014 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v014,
)
from taller_nlp import RetrieverHibridoRRF


class TestHiginioV014(unittest.TestCase):
    def test_parte_de_v012_y_solo_sustituye_el_retriever(self) -> None:
        v012 = crear_v012()
        v014 = crear_v014()
        retriever_v012 = v012.fabrica_herramientas.retriever
        retriever_v014 = v014.fabrica_herramientas.retriever

        self.assertEqual(v014.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v014.configuracion, v012.configuracion)
        self.assertEqual(v014.corpus, v012.corpus)
        self.assertIs(v014.esquema_respuesta, v012.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v014.middlewares),
            tuple(type(middleware) for middleware in v012.middlewares),
        )
        self.assertIs(v014.modelo, v012.modelo)
        self.assertEqual(
            dict(v014.control_peticiones.parametros),
            dict(v012.control_peticiones.parametros),
        )
        self.assertIs(v014.telemetria_auxiliar, v012.telemetria_auxiliar)
        self.assertEqual(
            v014.evaluador.k_retrieval,
            v012.evaluador.k_retrieval,
        )

        self.assertIsInstance(retriever_v014, RetrieverHibridoRRF)
        assert isinstance(retriever_v014, RetrieverHibridoRRF)
        self.assertEqual(retriever_v014.kk, KK_RRF)
        self.assertTrue(retriever_v014.aplica_filtros_metadatos)
        self.assertTrue(
            retriever_v014.retriever_denso.aplica_filtros_metadatos
        )
        self.assertIsNotNone(retriever_v012)
        assert retriever_v012 is not None
        self.assertEqual(
            retriever_v014.retriever_denso.nombre,
            retriever_v012.nombre,
        )


if __name__ == "__main__":
    unittest.main()
