from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from pydantic import ValidationError

from taller_nlp import (
    ConfiguracionAgente,
    ConstructorAgente,
    InformeEvaluacion,
    ManifiestoExperimento,
    DescriptorComponente,
    RespuestaAgente,
    ResultadoPregunta,
)

from tests.support import crear_corpus_temporal, crear_fabrica_prueba


def juez_determinista(respuesta: str, evidencias: tuple[str, ...]) -> bool:
    return bool(respuesta and evidencias)


class MiddlewarePrueba(AgentMiddleware):
    @property
    def parametros(self) -> dict[str, int]:
        return {"umbral": 2}


def crear_constructor(raiz: Path, *, k_retrieval: int = 3) -> ConstructorAgente:
    corpus, _ = crear_corpus_temporal(raiz / "corpus")
    configuracion = ConfiguracionAgente(
        modelo="modelo-prueba",
        system_prompt="Responde solo con fuentes verificables.",
        max_llamadas_total=6,
        limites_por_herramienta={"read_section": 1},
    )
    return ConstructorAgente(
        "equipo-a-v1",
        configuracion,
        corpus,
        crear_fabrica_prueba(corpus),
        juez_determinista,
        middlewares=(MiddlewarePrueba(),),
        k_retrieval=k_retrieval,
        tolerancia_absoluta=0.5,
        tolerancia_relativa=1e-5,
        numero_esperado=1,
    )


def crear_informe(ruta_golden: Path, *, nombre: str = "equipo-a-v1") -> InformeEvaluacion:
    resultado = ResultadoPregunta(
        id_pregunta="q1",
        familia="numerica",
        respuesta_agente=RespuestaAgente(
            respuesta="100 USD",
            cifra=100,
            unidad="USD",
            fuente="xbrl",
            latencia_ms=10,
        ),
        cifra_correcta=False,
        trayectoria_correcta=False,
    )
    return InformeEvaluacion(
        nombre_agente=nombre,
        ruta_jsonl=str(ruta_golden),
        resultados=(resultado,),
        k_retrieval=3,
        tolerancia_absoluta=0.5,
        tolerancia_relativa=1e-5,
        metodo_soporte_citas="juez_determinista",
    )


class TestManifiestoExperimento(unittest.TestCase):
    def test_rechaza_credenciales_en_parametros_declarados(self) -> None:
        with self.assertRaisesRegex(ValidationError, "credencial"):
            DescriptorComponente(
                referencia="mi_modulo.MiComponente",
                parametros={"api_key": "no-debe-persistirse"},
            )

    def test_captura_configuracion_componentes_entorno_y_codigo(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            constructor = crear_constructor(Path(temporal))
            manifiesto = ManifiestoExperimento.desde_constructor(
                constructor,
                revision_codigo="entrega-1",
                metadatos={"autor": "equipo-a"},
            )

            self.assertEqual(manifiesto.nombre_experimento, "equipo-a-v1")
            self.assertEqual(manifiesto.revision_codigo, "entrega-1")
            self.assertEqual(manifiesto.configuracion_agente.max_llamadas_total, 6)
            self.assertEqual(
                set(manifiesto.implementaciones_herramientas),
                {
                    "list_available",
                    "get_xbrl_fact",
                    "search_filings",
                    "read_section",
                },
            )
            self.assertEqual(
                manifiesto.middlewares_usuario[0].parametros,
                {"umbral": 2},
            )
            self.assertEqual(len(manifiesto.sha256_codigo), 64)
            self.assertIn("langchain", manifiesto.versiones_dependencias)
            self.assertIsNone(manifiesto.sha256_golden)

    def test_guarda_rutas_relativas_y_recarga_verificando_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            constructor = crear_constructor(raiz)
            golden = raiz / "datos" / "ciego.jsonl"
            golden.parent.mkdir()
            golden.write_text('{"id": "q1"}\n', encoding="utf-8")
            informe = crear_informe(golden)
            manifiesto = ManifiestoExperimento.desde_constructor(
                constructor,
                informe=informe,
            )
            ruta = raiz / "resultados" / "experimento.json"
            ruta.parent.mkdir()

            self.assertEqual(manifiesto.guardar(ruta), ruta.resolve())
            datos = json.loads(ruta.read_text(encoding="utf-8"))
            self.assertFalse(Path(datos["corpus"]["ruta_chunks"]).is_absolute())
            self.assertFalse(Path(datos["informe"]["ruta_jsonl"]).is_absolute())

            recargado = ManifiestoExperimento.cargar(ruta)
            self.assertEqual(recargado.nombre_experimento, manifiesto.nombre_experimento)
            self.assertEqual(recargado.sha256_golden, manifiesto.sha256_golden)
            self.assertEqual(
                recargado.corpus.ruta_chunks.resolve(),
                manifiesto.corpus.ruta_chunks.resolve(),
            )
            self.assertEqual(
                Path(recargado.informe.ruta_jsonl).resolve(),
                golden.resolve(),
            )

    def test_no_sobrescribe_un_manifiesto_existente(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            manifiesto = ManifiestoExperimento.desde_constructor(
                crear_constructor(raiz)
            )
            ruta = raiz / "experimento.json"
            manifiesto.guardar(ruta)
            with self.assertRaises(FileExistsError):
                manifiesto.guardar(ruta)

    def test_registra_hashes_de_un_indice_faiss_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(
                raiz / "corpus", con_faiss=True
            )
            configuracion = ConfiguracionAgente(
                modelo="modelo", system_prompt="prompt"
            )
            constructor = ConstructorAgente(
                "faiss-v1",
                configuracion,
                corpus,
                crear_fabrica_prueba(corpus),
                juez_determinista,
            )
            manifiesto = ManifiestoExperimento.desde_constructor(constructor)
            self.assertIsNotNone(manifiesto.sha256_indice)
            self.assertIsNotNone(manifiesto.sha256_metadatos_indice)
            self.assertEqual(manifiesto.comprobar_reproducibilidad(), ())

    def test_audita_el_entorno_y_los_artefactos_actuales(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            constructor = crear_constructor(raiz)
            manifiesto = ManifiestoExperimento.desde_constructor(constructor)
            self.assertEqual(manifiesto.comprobar_reproducibilidad(), ())

            constructor.corpus.ruta_chunks.write_text(
                "contenido modificado\n", encoding="utf-8"
            )
            diferencias = manifiesto.comprobar_reproducibilidad()
            self.assertTrue(
                any("artefacto chunks ha cambiado" in d for d in diferencias)
            )

    def test_detecta_modificacion_posterior_del_golden_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            constructor = crear_constructor(raiz)
            golden = raiz / "golden.jsonl"
            golden.write_text('{"id": "q1"}\n', encoding="utf-8")
            manifiesto = ManifiestoExperimento.desde_constructor(
                constructor,
                informe=crear_informe(golden),
            )
            ruta = raiz / "experimento.json"
            manifiesto.guardar(ruta)
            golden.write_text('{"id": "alterado"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "golden set no coincide"):
                ManifiestoExperimento.cargar(ruta)

    def test_rechaza_informe_de_otro_agente(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            constructor = crear_constructor(raiz)
            golden = raiz / "golden.jsonl"
            golden.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "agente distinto"):
                ManifiestoExperimento.desde_constructor(
                    constructor,
                    informe=crear_informe(golden, nombre="otro-agente"),
                )


if __name__ == "__main__":
    unittest.main()
