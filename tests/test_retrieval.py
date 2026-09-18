from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from taller_nlp import RetrieverFaiss
from taller_nlp.retrieval import extraer_chunk_ids_formateados

from tests.support import (
    CodificadorControlado,
    RetrieverControlado,
    como_recuperado,
    crear_corpus_temporal,
)


class TestContratoRetriever(unittest.TestCase):
    def test_normaliza_entrada_y_conserva_orden(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            resultados = tuple(como_recuperado(f, 1 - i / 10) for i, f in enumerate(fragmentos))
            retriever = RetrieverControlado(corpus, resultados)
            obtenidos = retriever.buscar("  risks  ", k=2)
            self.assertEqual(obtenidos, resultados)
            self.assertEqual(retriever.ultima_entrada["query"], "risks")

    def test_rechaza_salida_que_incumple_filtros_o_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            resultados = tuple(como_recuperado(f) for f in fragmentos)
            with self.assertRaisesRegex(ValueError, "para k=1"):
                RetrieverControlado(corpus, resultados).buscar("q", k=1)
            with self.assertRaisesRegex(ValueError, "ticker=OTHER"):
                RetrieverControlado(corpus, resultados[:1]).buscar(
                    "q", ticker="other"
                )

    def test_formato_es_estable_y_ids_se_pueden_extraer(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            retriever = RetrieverControlado(
                corpus, (como_recuperado(fragmentos[0], 0.75),)
            )
            texto = retriever.buscar_formateado("q")
            self.assertIn("[ACME-2024-1A-0000] ACME FY2024 Item 1A", texto)
            self.assertIn("puntuación 0.7500", texto)
            self.assertEqual(
                extraer_chunk_ids_formateados(texto),
                ("ACME-2024-1A-0000",),
            )


class TestRetrieverFaiss(unittest.TestCase):
    def test_ranking_prefijo_y_filtros_sin_descargar_modelo(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal), con_faiss=True)
            codificador = CodificadorControlado([1.0, 0.0])
            retriever = RetrieverFaiss(corpus, codificador=codificador)
            self.assertEqual(
                dict(retriever.parametros),
                {
                    "normalizar_embeddings": True,
                    "aplicar_filtros_metadatos": True,
                },
            )

            resultados = retriever.buscar("riesgo", k=2)
            self.assertEqual(
                [r.chunk_id for r in resultados],
                ["ACME-2024-1A-0000", "ACME-2023-7-0000"],
            )
            self.assertEqual(codificador.llamadas[0][0], ["query: riesgo"])
            self.assertTrue(codificador.llamadas[0][1])
            filtrado = retriever.buscar("riesgo", fiscal_year=2023, k=1)
            self.assertEqual(filtrado[0].fiscal_year, 2023)

    def test_puede_desactivar_filtros_para_medir_denso_plano(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal), con_faiss=True)
            retriever = RetrieverFaiss(
                corpus,
                codificador=CodificadorControlado([1.0, 0.0]),
                aplicar_filtros_metadatos=False,
            )

            resultados = retriever.buscar(
                "riesgo",
                fiscal_year=2023,
                k=1,
            )

            self.assertFalse(retriever.aplica_filtros_metadatos)
            self.assertEqual(resultados[0].fiscal_year, 2024)

    def test_rechaza_vector_de_dimension_incorrecta(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal), con_faiss=True)
            retriever = RetrieverFaiss(
                corpus, codificador=CodificadorControlado([1.0, 0.0, 0.0])
            )
            with self.assertRaisesRegex(ValueError, "se esperaba"):
                retriever.buscar("q")

    def test_exige_corpus_con_indice(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))
            with self.assertRaisesRegex(ValueError, "requiere"):
                RetrieverFaiss(corpus, codificador=CodificadorControlado([1, 0]))


if __name__ == "__main__":
    unittest.main()
