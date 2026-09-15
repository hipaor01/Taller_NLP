from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from taller_nlp import CorpusVariant

from tests.support import crear_corpus_temporal


class TestCorpusVariant(unittest.TestCase):
    def test_detecta_modificacion_de_un_artefacto(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz)
            corpus.ruta_chunks.write_text("contenido alterado\n", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "SHA-256"):
                CorpusVariant.model_validate(corpus.model_dump())

    def test_manifiesto_resuelve_rutas_relativas_desde_su_directorio(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "datos")
            ruta_manifiesto = raiz / "manifiesto.json"
            ruta_manifiesto.write_text(
                json.dumps(corpus.datos_manifiesto(raiz)), encoding="utf-8"
            )
            recargado = CorpusVariant.cargar_manifiesto(ruta_manifiesto)
            self.assertEqual(
                recargado.ruta_secciones.resolve(),
                corpus.ruta_secciones.resolve(),
            )
            self.assertEqual(recargado.ruta_xbrl.resolve(), corpus.ruta_xbrl.resolve())
            self.assertEqual(
                recargado.ruta_chunks.resolve(), corpus.ruta_chunks.resolve()
            )
            self.assertEqual(recargado.nombre, corpus.nombre)
            self.assertEqual(recargado.sha256_chunks, corpus.sha256_chunks)
            self.assertTrue(recargado.ruta_chunks.is_absolute())


if __name__ == "__main__":
    unittest.main()
