from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from experimentos.higinio.agente_v004 import crear_constructor as crear_v004
from experimentos.higinio.agente_v006 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v006,
)
from experimentos.higinio.verificador_xbrl import VerificadorCifrasXBRL
from taller_nlp import RetrieverConReescritura, RetrieverHibridoRRF


class TestHiginioV006(unittest.TestCase):
    def test_parte_de_v004_y_solo_anade_el_verificador(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            v004 = crear_v004()
            v006 = crear_v006()

        retriever_v004 = v004.fabrica_herramientas.retriever
        retriever_v006 = v006.fabrica_herramientas.retriever

        self.assertEqual(v006.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v006.configuracion, v004.configuracion)
        self.assertEqual(v006.corpus, v004.corpus)
        self.assertEqual(v006.evaluador.k_retrieval, v004.evaluador.k_retrieval)
        self.assertEqual(
            v006.evaluador.tolerancia_relativa,
            v004.evaluador.tolerancia_relativa,
        )
        self.assertIsInstance(retriever_v004, RetrieverConReescritura)
        self.assertIsInstance(retriever_v006, RetrieverConReescritura)
        assert isinstance(retriever_v004, RetrieverConReescritura)
        assert isinstance(retriever_v006, RetrieverConReescritura)
        self.assertEqual(retriever_v006.nombre, retriever_v004.nombre)
        self.assertEqual(
            dict(retriever_v006.parametros),
            dict(retriever_v004.parametros),
        )
        self.assertIsInstance(
            retriever_v006.retriever_base,
            RetrieverHibridoRRF,
        )
        self.assertEqual(
            dict(retriever_v006.retriever_base.parametros),
            dict(retriever_v004.retriever_base.parametros),
        )
        self.assertIsNotNone(v006.telemetria_auxiliar)
        self.assertEqual(len(v006.middlewares), 1)
        self.assertIsInstance(v006.middlewares[0], VerificadorCifrasXBRL)


if __name__ == "__main__":
    unittest.main()
