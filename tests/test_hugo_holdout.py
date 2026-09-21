"""Tests del plan B de Hugo para las preguntas ciegas (experimentos/hugo).

No prueban código común: prueban ``experimentos/hugo/holdout.py`` y
``experimentos/hugo/evaluar_holdout.py``, que solo usan sus clases.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experimentos.baseline import crear_constructor_baseline, crear_corpus_baseline
from experimentos.hugo import evaluar_holdout as cli
from experimentos.hugo.holdout import cargar_holdout, evaluar_holdout
from taller_nlp import LlamadaHerramienta, RespuestaAgente, RetrieverFaiss
from taller_nlp.golden import CasoGolden


VALIDA_NUMERICA = {
    "id": "h-01",
    "pregunta": "¿Cuál fue el revenue de NVIDIA en FY2024?",
    "familia": "numerica",
    "ticker": "NVDA",
    "fiscal_year": 2024,
    "respuesta_esperada": "60.922 millones de dólares",
    "cifra_esperada": 60922000000.0,
    "unidad": "USD",
    "concept_xbrl": "Revenues",
    "item_esperado": None,
    "ancla_texto": None,
    "ancla_inicio": None,
    "ancla_fin": None,
    "chunk_id_esperado": None,
    "herramienta_esperada": ["get_xbrl_fact"],
    "autor": "prueba",
}
CONCEPTO_NO_REPORTADO = {
    **VALIDA_NUMERICA,
    "id": "h-02",
    "pregunta": "¿Cuál fue el beneficio bruto de Amazon en FY2025?",
    "ticker": "AMZN",
    "fiscal_year": 2025,
    "concept_xbrl": "GrossProfit",
    "cifra_esperada": 0.0,
    "respuesta_esperada": "Amazon no reporta GrossProfit en us-gaap.",
}
EMPRESA_FUERA = {
    **VALIDA_NUMERICA,
    "id": "h-03",
    "pregunta": "¿Cuál fue el revenue de Tesla en FY2025?",
    "ticker": "TSLA",
    "fiscal_year": 2025,
}
SIN_CIFRA = {
    **VALIDA_NUMERICA,
    "id": "h-04",
    "pregunta": "¿Cuál fue el pasivo total de Amazon en FY2025?",
    "ticker": "AMZN",
    "fiscal_year": 2025,
    "concept_xbrl": "Liabilities",
    "cifra_esperada": None,
    "unidad": None,
    "respuesta_esperada": "No está en el corpus.",
}
DESCONOCIDA = {"id": "h-05", "pregunta": "¿Qué opina NVIDIA del clima?", "familia": "otra"}


def _escribir(lineas: list[str]) -> Path:
    ruta = Path(tempfile.mkdtemp()) / "holdout.jsonl"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def _jsonl(*casos: dict) -> Path:
    return _escribir([json.dumps(c, ensure_ascii=False) for c in casos])


def _respuesta(fuente: str, cifra: float | None = None, unidad: str | None = None) -> RespuestaAgente:
    return RespuestaAgente(
        respuesta="respuesta de prueba",
        cifra=cifra,
        unidad=unidad,
        fuente=fuente,
        llamadas=(
            LlamadaHerramienta(
                id="1",
                nombre="get_xbrl_fact",
                argumentos={"ticker": "NVDA", "fiscal_year": 2024, "concept": "Revenues"},
                duracion_ms=1,
                resultado="ok",
            ),
        ),
        latencia_ms=1000,
        coste_usd=0.001,
    )


class _Motor:
    def __init__(self, respuestas: dict[str, RespuestaAgente] | None = None) -> None:
        self.respuestas = respuestas or {}
        self.preguntas: list[str] = []

    def responder(self, pregunta: str) -> RespuestaAgente:
        self.preguntas.append(pregunta)
        return self.respuestas.get(pregunta, _respuesta("ninguna"))


class TestClasificacion(unittest.TestCase):
    def test_el_cargador_comun_rechaza_el_conjunto_entero(self) -> None:
        # El motivo de este plan B: con una sola pregunta sin respuesta, el
        # cargador común no deja evaluar ninguna.
        with self.assertRaises(ValueError):
            CasoGolden.cargar_jsonl(
                _jsonl(VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO), crear_corpus_baseline()
            )

    def test_clasifica_cada_linea_sin_rechazar_el_conjunto(self) -> None:
        ruta = _escribir(
            [json.dumps(c, ensure_ascii=False) for c in (
                VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO, EMPRESA_FUERA, SIN_CIFRA, DESCONOCIDA
            )]
            + ["{esto no es json"]
        )
        tipos = {c.id: c.tipo for c in cargar_holdout(ruta, crear_corpus_baseline())}
        # Desde c6393b4 el cargador común acepta una numérica sin cifra ni
        # unidad (espera_ausencia_numerica) y el evaluador común la puntúa:
        # entonces h-04 ya es un caso golden normal.
        sin_cifra = (
            "golden"
            if hasattr(CasoGolden, "espera_ausencia_numerica")
            else "sin_respuesta"
        )
        self.assertEqual(
            tipos,
            {
                "h-01": "golden",
                "h-02": "sin_respuesta",
                "h-03": "sin_respuesta",
                "h-04": sin_cifra,
                "h-05": "no_evaluable",
                "linea-6": "no_evaluable",
            },
        )


class TestEvaluacion(unittest.TestCase):
    def setUp(self) -> None:
        self.constructor = crear_constructor_baseline()

    def _evaluar(self, motor: _Motor, *casos: dict, progreso: Path | None = None):
        return evaluar_holdout(
            cargar_holdout(_jsonl(*casos), self.constructor.corpus),
            evaluador=self.constructor.evaluador,
            responder=motor.responder,
            ruta_progreso=progreso,
        )

    def test_puntua_cada_tipo(self) -> None:
        motor = _Motor(
            {
                VALIDA_NUMERICA["pregunta"]: _respuesta("xbrl", 60922000000.0, "USD"),
                CONCEPTO_NO_REPORTADO["pregunta"]: _respuesta("ninguna"),
                EMPRESA_FUERA["pregunta"]: _respuesta("texto", 97690000000.0, "USD"),
            }
        )
        tabla = self._evaluar(motor, VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO, EMPRESA_FUERA, DESCONOCIDA)
        filas = {f["id"]: f for f in tabla.to_dict("records")}
        self.assertEqual(len(motor.preguntas), 4)
        self.assertTrue(filas["h-01"]["acierto"])
        self.assertTrue(filas["h-01"]["cifra"])
        self.assertTrue(filas["h-02"]["acierto"])
        self.assertFalse(filas["h-03"]["acierto"], "Inventar una cifra debe contar como fallo.")
        self.assertIsNone(filas["h-05"]["acierto"])

    def test_reanuda_sin_repetir_preguntas(self) -> None:
        progreso = Path(tempfile.mkdtemp()) / "progreso.jsonl"
        primero = _Motor()
        self._evaluar(primero, VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO, progreso=progreso)
        segundo = _Motor()
        self._evaluar(segundo, VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO, progreso=progreso)
        self.assertEqual(len(primero.preguntas), 2)
        self.assertEqual(segundo.preguntas, [])

    def test_un_fallo_del_motor_no_detiene_el_resto(self) -> None:
        class _Falla(_Motor):
            def responder(self, pregunta: str) -> RespuestaAgente:
                self.preguntas.append(pregunta)
                if pregunta == CONCEPTO_NO_REPORTADO["pregunta"]:
                    raise RuntimeError("fallo de red simulado")
                return _respuesta("ninguna")

        tabla = self._evaluar(_Falla(), CONCEPTO_NO_REPORTADO, EMPRESA_FUERA)
        fila = tabla.set_index("id").loc["h-02"]
        self.assertEqual(len(tabla), 2)
        self.assertIn("fallo de red simulado", fila["error"])
        self.assertFalse(fila["acierto"])


class _Espia:
    def __init__(self) -> None:
        self.dentro = 0
        self.maximo = 0
        self._cerrojo = threading.Lock()

    def encode(self, textos, normalize_embeddings=True, convert_to_numpy=True):
        with self._cerrojo:
            self.dentro += 1
            self.maximo = max(self.maximo, self.dentro)
        time.sleep(0.05)
        with self._cerrojo:
            self.dentro -= 1
        vector = np.ones((1, 384), dtype="float32")
        return vector / np.linalg.norm(vector)


class TestLineaDeComandos(unittest.TestCase):
    def setUp(self) -> None:
        self._original = RetrieverFaiss._codificar

    def tearDown(self) -> None:
        RetrieverFaiss._codificar = self._original  # type: ignore[method-assign]

    def test_cerrojo_evita_codificaciones_simultaneas(self) -> None:
        cli.activar_cerrojo_codificacion()
        cli.activar_cerrojo_codificacion()  # idempotente
        espia = _Espia()
        retriever = RetrieverFaiss(crear_corpus_baseline(), codificador=espia, aplicar_filtros_metadatos=True)
        with ThreadPoolExecutor(max_workers=4) as hilos:
            list(hilos.map(lambda q: retriever.buscar(q, ticker="NVDA", k=3), "abcd"))
        self.assertEqual(espia.maximo, 1)

    def test_ejecuta_de_principio_a_fin_con_cualquier_variante(self) -> None:
        motor = _Motor({VALIDA_NUMERICA["pregunta"]: _respuesta("xbrl", 60922000000.0, "USD")})
        base = crear_constructor_baseline()
        falsa = SimpleNamespace(
            crear_constructor=lambda: SimpleNamespace(
                nombre="variante-de-prueba",
                corpus=base.corpus,
                evaluador=base.evaluador,
                construir_motor=lambda: motor,
            )
        )
        directorio = Path(tempfile.mkdtemp())
        with patch.object(cli, "import_module", return_value=falsa):
            codigo = cli.main(
                [
                    str(_jsonl(VALIDA_NUMERICA, CONCEPTO_NO_REPORTADO, SIN_CIFRA)),
                    "--variante", "experimentos.higinio.agente_v009",
                    "--salida", str(directorio / "holdout.csv"),
                    "--progreso", str(directorio / "progreso.jsonl"),
                ]
            )
        self.assertEqual(codigo, 0)
        self.assertTrue((directorio / "holdout.csv").is_file())
        self.assertTrue((directorio / "holdout.json").is_file())
        self.assertEqual(len(motor.preguntas), 3)


if __name__ == "__main__":
    unittest.main()
