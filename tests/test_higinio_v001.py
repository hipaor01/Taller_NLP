from __future__ import annotations

import unittest

from experimentos.baseline import crear_constructor_baseline
from experimentos.higinio.agente_v001 import (
    NOMBRE_VARIANTE,
    crear_constructor,
)


class TestHiginioV001(unittest.TestCase):
    def test_solo_activa_los_filtros_del_retriever_baseline(self) -> None:
        baseline = crear_constructor_baseline()
        variante = crear_constructor()
        retriever_baseline = baseline.fabrica_herramientas.retriever
        retriever_variante = variante.fabrica_herramientas.retriever

        self.assertEqual(variante.nombre, NOMBRE_VARIANTE)
        self.assertEqual(variante.configuracion, baseline.configuracion)
        self.assertEqual(variante.corpus, baseline.corpus)
        self.assertEqual(
            variante.evaluador.k_retrieval,
            baseline.evaluador.k_retrieval,
        )
        self.assertEqual(
            variante.evaluador.tolerancia_absoluta,
            baseline.evaluador.tolerancia_absoluta,
        )
        self.assertEqual(
            variante.evaluador.tolerancia_relativa,
            baseline.evaluador.tolerancia_relativa,
        )
        self.assertIsNotNone(retriever_baseline)
        self.assertIsNotNone(retriever_variante)
        assert retriever_baseline is not None
        assert retriever_variante is not None
        self.assertFalse(retriever_baseline.aplica_filtros_metadatos)
        self.assertTrue(retriever_variante.aplica_filtros_metadatos)
        self.assertEqual(
            retriever_variante.parametros["normalizar_embeddings"],
            retriever_baseline.parametros["normalizar_embeddings"],
        )

    def test_la_descripcion_de_search_filings_declara_filtros_reales(self) -> None:
        constructor = crear_constructor()
        descripcion = constructor.fabrica_herramientas.descripciones[
            "search_filings"
        ]

        self.assertNotIn("no restringen el ranking denso", descripcion)


if __name__ == "__main__":
    unittest.main()
