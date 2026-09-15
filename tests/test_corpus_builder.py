from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taller_nlp import ConstructorCorpusVariant, CorpusVariant, IndexadorCorpus

from tests.support import (
    IndexadorTexto,
    TroceadorSeccionCompleta,
    crear_corpus_temporal,
)


class IndexadorFallido(IndexadorCorpus):
    def __init__(self) -> None:
        super().__init__("fallido", "test")

    def _construir_indice(self, fragmentos, directorio_salida):
        del fragmentos
        (directorio_salida / "parcial.txt").write_text("parcial", encoding="utf-8")
        raise RuntimeError("fallo deliberado")


class TestConstructorCorpusVariant(unittest.TestCase):
    def test_construye_publica_y_recarga_manifiesto_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            fuente, _ = crear_corpus_temporal(raiz / "fuente")
            indexador = IndexadorTexto()
            constructor = ConstructorCorpusVariant(
                fuente,
                TroceadorSeccionCompleta(),
                indexador=indexador,
            )
            destino = raiz / "variante"
            variante = constructor.construir("equipo-a-v1", destino)

            self.assertTrue((destino / "chunks.jsonl").is_file())
            manifiesto = destino / "corpus_variant.json"
            self.assertTrue(manifiesto.is_file())
            datos = json.loads(manifiesto.read_text(encoding="utf-8"))
            self.assertFalse(Path(datos["ruta_chunks"]).is_absolute())
            self.assertEqual(CorpusVariant.cargar_manifiesto(manifiesto), variante)
            self.assertEqual(variante.troceador, "seccion-completa")
            self.assertEqual(variante.artefactos_indice.nombre_indexador, "indice-texto")

    def test_no_sobrescribe_destino_existente(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            fuente, _ = crear_corpus_temporal(raiz / "fuente")
            destino = raiz / "existente"
            destino.mkdir()
            constructor = ConstructorCorpusVariant(
                fuente, TroceadorSeccionCompleta()
            )
            with self.assertRaises(FileExistsError):
                constructor.construir("variante", destino)

    def test_un_fallo_no_publica_destino_ni_deja_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            fuente, _ = crear_corpus_temporal(raiz / "fuente")
            constructor = ConstructorCorpusVariant(
                fuente,
                TroceadorSeccionCompleta(),
                indexador=IndexadorFallido(),
            )
            destino = raiz / "variante"
            with self.assertRaisesRegex(RuntimeError, "fallo deliberado"):
                constructor.construir("variante", destino)
            self.assertFalse(destino.exists())
            self.assertEqual(list(raiz.glob(".construccion-corpus-*")), [])


if __name__ == "__main__":
    unittest.main()
