from __future__ import annotations

import unittest

from experimentos.higinio.agente_v002 import KK_RRF
from experimentos.higinio.agente_v013 import crear_constructor as crear_v013
from experimentos.higinio.agente_v015 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v015,
)
from taller_nlp import RetrieverHibridoRRF


class TestHiginioV015(unittest.TestCase):
    def test_parte_de_v013_y_solo_sustituye_el_retriever(self) -> None:
        v013 = crear_v013()
        v015 = crear_v015()
        retriever_v013 = v013.fabrica_herramientas.retriever
        retriever_v015 = v015.fabrica_herramientas.retriever

        self.assertEqual(v015.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v015.configuracion, v013.configuracion)
        self.assertEqual(v015.corpus, v013.corpus)
        self.assertIs(v015.esquema_respuesta, v013.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v015.middlewares),
            tuple(type(middleware) for middleware in v013.middlewares),
        )
        self.assertIs(v015.modelo, v013.modelo)
        self.assertEqual(
            dict(v015.control_peticiones.parametros),
            dict(v013.control_peticiones.parametros),
        )
        self.assertIs(v015.telemetria_auxiliar, v013.telemetria_auxiliar)
        self.assertEqual(
            v015.evaluador.k_retrieval,
            v013.evaluador.k_retrieval,
        )

        self.assertIsInstance(retriever_v015, RetrieverHibridoRRF)
        assert isinstance(retriever_v015, RetrieverHibridoRRF)
        self.assertEqual(retriever_v015.kk, KK_RRF)
        self.assertTrue(retriever_v015.aplica_filtros_metadatos)
        self.assertTrue(
            retriever_v015.retriever_denso.aplica_filtros_metadatos
        )
        self.assertIsNotNone(retriever_v013)
        assert retriever_v013 is not None
        self.assertEqual(
            retriever_v015.retriever_denso.nombre,
            retriever_v013.nombre,
        )


if __name__ == "__main__":
    unittest.main()
