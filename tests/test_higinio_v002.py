from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experimentos.higinio.agente_v001 import crear_constructor as crear_v001
from experimentos.higinio.agente_v002 import (
    KK_RRF,
    NOMBRE_VARIANTE,
    crear_constructor as crear_v002,
)
from taller_nlp import RetrieverHibridoRRF
from tests.support import (
    RetrieverControlado,
    como_recuperado,
    crear_corpus_temporal,
)


class _BM25Controlado:
    def __init__(self, puntuaciones: list[float]) -> None:
        self.puntuaciones = puntuaciones
        self.consultas: list[list[str]] = []

    def get_scores(self, consulta: list[str]) -> list[float]:
        self.consultas.append(consulta)
        return self.puntuaciones


class TestRetrieverHibridoRRF(unittest.TestCase):
    def test_fusiona_posiciones_sin_sumar_puntuaciones(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            denso = RetrieverControlado(
                corpus,
                tuple(
                    como_recuperado(fragmento, puntuacion)
                    for fragmento, puntuacion in zip(
                        fragmentos, (0.99, 0.10), strict=True
                    )
                ),
            )
            bm25 = _BM25Controlado([0.0, 20.0])
            retriever = RetrieverHibridoRRF(
                corpus,
                retriever_denso=denso,
                indice_bm25=bm25,
                fragmentos_bm25=fragmentos,
                tokenizador=str.split,
                kk=60,
            )

            resultados = retriever.buscar("supply risks", k=2)

            self.assertEqual(bm25.consultas, [["supply", "risks"]])
            self.assertEqual(
                [resultado.chunk_id for resultado in resultados],
                [fragmentos[0].chunk_id, fragmentos[1].chunk_id],
            )
            esperado = 1 / 61 + 1 / 62
            self.assertAlmostEqual(resultados[0].puntuacion, esperado)
            self.assertAlmostEqual(resultados[1].puntuacion, esperado)
            self.assertEqual(
                resultados[0].detalles_puntuacion["posicion_dense"], 1.0
            )
            self.assertEqual(
                resultados[0].detalles_puntuacion["posicion_bm25"], 2.0
            )
            self.assertEqual(
                resultados[1].detalles_puntuacion["posicion_dense"], 2.0
            )
            self.assertEqual(
                resultados[1].detalles_puntuacion["posicion_bm25"], 1.0
            )


class TestHiginioV002(unittest.TestCase):
    def test_acumula_bm25_rrf_sobre_el_retriever_filtrado_de_v001(self) -> None:
        v001 = crear_v001()
        v002 = crear_v002()
        retriever_v001 = v001.fabrica_herramientas.retriever
        retriever_v002 = v002.fabrica_herramientas.retriever

        self.assertEqual(v002.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v002.configuracion, v001.configuracion)
        self.assertEqual(v002.corpus, v001.corpus)
        self.assertEqual(v002.evaluador.k_retrieval, v001.evaluador.k_retrieval)
        self.assertIsInstance(retriever_v002, RetrieverHibridoRRF)
        assert isinstance(retriever_v002, RetrieverHibridoRRF)
        self.assertTrue(retriever_v002.aplica_filtros_metadatos)
        self.assertTrue(retriever_v002.retriever_denso.aplica_filtros_metadatos)
        self.assertEqual(retriever_v002.kk, KK_RRF)
        self.assertIsNotNone(retriever_v001)
        assert retriever_v001 is not None
        self.assertEqual(
            retriever_v002.retriever_denso.nombre,
            retriever_v001.nombre,
        )


if __name__ == "__main__":
    unittest.main()
