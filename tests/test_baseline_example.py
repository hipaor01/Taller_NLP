from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

import faiss

from experimentos.baseline import (
    MODELO_AGENTE,
    SYSTEM_PROMPT,
    crear_configuracion_baseline,
    crear_constructor_baseline,
    crear_corpus_baseline,
)
from taller_nlp import ProgresoConsolaMiddleware


class TestEjemploBaseline(unittest.TestCase):
    def test_declara_y_valida_los_artefactos_docentes(self) -> None:
        corpus = crear_corpus_baseline()
        self.assertEqual(corpus.nombre, "baseline-notebook-s1")
        self.assertEqual(corpus.dimension_embeddings, 384)
        indice = faiss.read_index(str(corpus.ruta_indice_faiss))
        self.assertEqual(indice.ntotal, 1749)

    def test_conserva_modelo_prompt_y_reglas_del_notebook(self) -> None:
        configuracion = crear_configuracion_baseline()
        self.assertEqual(configuracion.modelo, MODELO_AGENTE)
        self.assertEqual(configuracion.system_prompt, SYSTEM_PROMPT.strip())
        self.assertIn("Para cualquier CIFRA", configuracion.system_prompt)
        self.assertIn("Cita el chunk_id", configuracion.system_prompt)
        self.assertEqual(configuracion.max_iteraciones, 6)
        self.assertEqual(configuracion.timeout_s, 60)
        self.assertEqual(configuracion.solicitudes_modelo_por_minuto, 18)
        self.assertEqual(configuracion.max_reintentos_rate_limit, 3)

    def test_ensambla_sin_api_ni_descargar_embeddings(self) -> None:
        constructor = crear_constructor_baseline(juez_citas=lambda *_: True)
        self.assertEqual(constructor.nombre, "baseline-notebook-s1")
        self.assertIs(
            constructor.fabrica_herramientas.retriever.corpus,
            constructor.corpus,
        )
        self.assertEqual(
            dict(constructor.fabrica_herramientas.retriever.parametros),
            {"normalizar_embeddings": True},
        )

    def test_middleware_hace_visibles_modelo_y_tool(self) -> None:
        middleware = ProgresoConsolaMiddleware()
        salida = io.StringIO()
        with redirect_stderr(salida):
            self.assertEqual(
                middleware.wrap_model_call(object(), lambda _: "modelo-ok"),
                "modelo-ok",
            )
            peticion = SimpleNamespace(
                tool_call={"name": "list_available", "args": {}}
            )
            self.assertEqual(
                middleware.wrap_tool_call(peticion, lambda _: "tool-ok"),
                "tool-ok",
            )
        texto = salida.getvalue()
        self.assertIn("[baseline]", texto)
        self.assertIn("Llamada 1 al modelo", texto)
        self.assertIn("Tool list_available", texto)

    def test_middleware_admite_etiqueta_de_otro_experimento(self) -> None:
        middleware = ProgresoConsolaMiddleware("reranking")
        salida = io.StringIO()
        with redirect_stderr(salida):
            middleware.wrap_model_call(object(), lambda _: "ok")

        self.assertIn("[reranking] Llamada 1 al modelo", salida.getvalue())
        self.assertEqual(middleware.parametros["etiqueta"], "reranking")


if __name__ == "__main__":
    unittest.main()
