from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from experimentos.higinio.agente_v002 import crear_constructor as crear_v002
from experimentos.higinio.agente_v004 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v004,
)
from taller_nlp import RetrieverConReescritura, RetrieverHibridoRRF


class TestHiginioV004(unittest.TestCase):
    def test_aplica_reescritura_sobre_el_hibrido_completo_de_v002(self) -> None:
        v002 = crear_v002()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            v004 = crear_v004()
        retriever_v002 = v002.fabrica_herramientas.retriever
        retriever_v004 = v004.fabrica_herramientas.retriever

        self.assertEqual(v004.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v004.configuracion, v002.configuracion)
        self.assertEqual(v004.corpus, v002.corpus)
        self.assertEqual(v004.evaluador.k_retrieval, v002.evaluador.k_retrieval)
        self.assertIsInstance(retriever_v004, RetrieverConReescritura)
        assert isinstance(retriever_v004, RetrieverConReescritura)
        self.assertIsInstance(
            retriever_v004.retriever_base,
            RetrieverHibridoRRF,
        )
        self.assertIsNotNone(retriever_v002)
        assert retriever_v002 is not None
        self.assertEqual(
            retriever_v004.retriever_base.nombre,
            retriever_v002.nombre,
        )
        self.assertEqual(
            dict(retriever_v004.retriever_base.parametros),
            dict(retriever_v002.parametros),
        )
        hibrido = retriever_v004.retriever_base
        assert isinstance(hibrido, RetrieverHibridoRRF)
        self.assertTrue(hibrido.retriever_denso.aplica_filtros_metadatos)
        self.assertFalse(retriever_v004.reescritor.modelo_disponible)
        self.assertIsNotNone(v004.telemetria_auxiliar)


if __name__ == "__main__":
    unittest.main()
