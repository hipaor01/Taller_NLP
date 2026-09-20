from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from taller_nlp import (
    CasoGolden,
    EvaluadorFinanciero,
    LlamadaHerramienta,
    RateLimitAgotadoError,
    RespuestaAgente,
)

from tests.support import ANCLA_2024, TEXTO_2024, crear_corpus_temporal, escribir_jsonl


def caso_extractivo() -> dict:
    inicio = TEXTO_2024.index(ANCLA_2024)
    return {
        "id": "ext-001",
        "pregunta": "¿Qué riesgo se describe?",
        "familia": "extractiva",
        "ticker": "ACME",
        "fiscal_year": 2024,
        "respuesta_esperada": "Las disrupciones podrían dañar las operaciones.",
        "item_esperado": "1A",
        "ancla_texto": ANCLA_2024,
        "ancla_inicio": inicio,
        "ancla_fin": inicio + len(ANCLA_2024),
        "chunk_id_esperado": "ACME-2024-1A-0000",
        "herramienta_esperada": ["search_filings"],
        "autor": "equipo",
    }


def caso_numerico() -> dict:
    return {
        "id": "num-001",
        "pregunta": "¿Cuál fue el beneficio neto?",
        "familia": "numerica",
        "ticker": "ACME",
        "fiscal_year": 2024,
        "respuesta_esperada": "100 USD",
        "cifra_esperada": 100.0,
        "unidad": "USD",
        "concept_xbrl": "NetIncomeLoss",
        "herramienta_esperada": ["get_xbrl_fact"],
        "autor": "equipo",
    }


def caso_numerico_sin_respuesta() -> dict:
    caso = caso_numerico()
    caso.update(
        {
            "id": "num-ausente-001",
            "pregunta": "¿Cuál fue el beneficio bruto?",
            "respuesta_esperada": "El dato no está disponible en el corpus.",
            "cifra_esperada": None,
            "unidad": None,
            "concept_xbrl": "GrossProfit",
        }
    )
    return caso


def caso_comparativo() -> dict:
    inicio = TEXTO_2024.index(ANCLA_2024)
    return {
        "id": "comp-001",
        "pregunta": (
            "¿Cómo cambió el beneficio neto entre 2023 y 2024 y por qué?"
        ),
        "familia": "comparativa",
        "ticker": "ACME",
        "fiscal_year": 2024,
        "respuesta_esperada": (
            "Aumentó de 80 a 100 USD; las disrupciones afectaron las operaciones."
        ),
        "cifra_esperada": 100.0,
        "unidad": "USD",
        "concept_xbrl": "NetIncomeLoss",
        "item_esperado": "1A",
        "ancla_texto": ANCLA_2024,
        "ancla_inicio": inicio,
        "ancla_fin": inicio + len(ANCLA_2024),
        "chunk_id_esperado": "ACME-2024-1A-0000",
        "herramienta_esperada": ["get_xbrl_fact", "search_filings"],
        "autor": "equipo",
    }


class TestCasoGolden(unittest.TestCase):
    def test_carga_jsonl_con_nombre_arbitrario_y_valida_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "preguntas_ciegas_17.jsonl"
            escribir_jsonl(ruta, [caso_extractivo(), caso_numerico()])
            casos = CasoGolden.cargar_jsonl(ruta, corpus, numero_esperado=2)
            self.assertEqual([caso.id for caso in casos], ["ext-001", "num-001"])

    def test_rechaza_ancla_larga_y_offsets_incoherentes(self) -> None:
        largo = caso_extractivo()
        largo["ancla_texto"] = " ".join(["palabra"] * 41)
        largo["ancla_fin"] = largo["ancla_inicio"] + len(largo["ancla_texto"])
        with self.assertRaisesRegex(ValidationError, "máximo es 40"):
            CasoGolden.model_validate(largo)

        incoherente = caso_extractivo()
        incoherente["ancla_fin"] += 1
        with self.assertRaisesRegex(ValidationError, "longitud del ancla"):
            CasoGolden.model_validate(incoherente)

    def test_informa_ids_repetidos_y_minimo_de_comparativas(self) -> None:
        import pandas as pd

        problemas = CasoGolden.validar_registros(
            [caso_numerico(), caso_numerico()],
            secciones=pd.DataFrame(
                [{"ticker": "ACME", "fiscal_year": 2024, "item": "1A", "texto": TEXTO_2024}]
            ),
            xbrl=pd.DataFrame(
                [{"ticker": "ACME", "fiscal_year": 2024, "concept": "NetIncomeLoss"}]
            ),
            minimo_comparativas=1,
        )
        self.assertTrue(any("id repetido" in p for p in problemas))
        self.assertTrue(any("hacen falta 1 comparativas" in p for p in problemas))

    def test_acepta_magnitud_y_ejercicio_ausentes(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            magnitud_ausente = caso_numerico_sin_respuesta()
            ejercicio_ausente = caso_numerico_sin_respuesta()
            ejercicio_ausente.update(
                {
                    "id": "num-ejercicio-ausente-001",
                    "fiscal_year": 2022,
                    "concept_xbrl": "NetIncomeLoss",
                }
            )
            ruta = raiz / "ausentes.jsonl"
            escribir_jsonl(ruta, [magnitud_ausente, ejercicio_ausente])

            casos = CasoGolden.cargar_jsonl(ruta, corpus, numero_esperado=2)

            self.assertTrue(all(caso.espera_ausencia_numerica for caso in casos))

    def test_rechaza_ausencia_incoherente_o_parcial(self) -> None:
        import pandas as pd

        parcial = caso_numerico_sin_respuesta()
        parcial["unidad"] = "USD"
        with self.assertRaisesRegex(ValidationError, "ambas informadas"):
            CasoGolden.model_validate(parcial)

        falso_ausente = caso_numerico_sin_respuesta()
        falso_ausente["concept_xbrl"] = "NetIncomeLoss"
        problemas = CasoGolden.validar_registros(
            [falso_ausente],
            secciones=pd.DataFrame(
                [
                    {
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "item": "1A",
                        "texto": TEXTO_2024,
                    }
                ]
            ),
            xbrl=pd.DataFrame(
                [
                    {
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "concept": "NetIncomeLoss",
                    }
                ]
            ),
        )

        self.assertTrue(any("sí reporta" in problema for problema in problemas))


class TestEvaluadorFinanciero(unittest.TestCase):
    def test_puntua_la_abstencion_numerica_correcta(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "ausente.jsonl"
            escribir_jsonl(ruta, [caso_numerico_sin_respuesta()])
            llamada = LlamadaHerramienta(
                id="xbrl-ausente",
                nombre="get_xbrl_fact",
                argumentos={
                    "ticker": "ACME",
                    "fiscal_year": 2024,
                    "concept": "GrossProfit",
                },
                duracion_ms=0,
                resultado="ACME no reportó 'GrossProfit' en FY2024.",
            )
            respuesta = RespuestaAgente(
                respuesta="El dato no está disponible en el corpus.",
                fuente="ninguna",
                llamadas=(llamada,),
                latencia_ms=1,
            )

            informe = EvaluadorFinanciero(corpus).evaluar(
                nombre_agente="agente-prudente",
                responder=lambda _: respuesta,
                ruta_jsonl=ruta,
            )

            resultado = informe.resultados[0]
            self.assertTrue(resultado.cifra_correcta)
            self.assertTrue(resultado.trayectoria_correcta)
            self.assertTrue(resultado.acierto)
            self.assertEqual(resultado.observaciones, ())

    def test_rechaza_falsa_abstencion_o_error_tecnico(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal) / "corpus")
            evaluador = EvaluadorFinanciero(corpus)
            caso = CasoGolden.model_validate(caso_numerico_sin_respuesta())
            base = RespuestaAgente(
                respuesta="El dato no está disponible.",
                fuente="ninguna",
                latencia_ms=1,
            )
            fuente_incorrecta = base.model_copy(update={"fuente": "texto"})
            cifra_inventada = base.model_copy(
                update={"cifra": 100.0, "unidad": "USD", "fuente": "xbrl"}
            )
            error_tecnico = RespuestaAgente(
                respuesta="",
                fuente="ninguna",
                latencia_ms=0,
                error="fallo de validación",
            )

            self.assertTrue(evaluador._cifra_correcta(caso, base))
            self.assertFalse(evaluador._cifra_correcta(caso, fuente_incorrecta))
            self.assertFalse(evaluador._cifra_correcta(caso, cifra_inventada))
            self.assertFalse(evaluador._cifra_correcta(caso, error_tecnico))

    def test_rechaza_cita_textual_que_no_aparece_en_el_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal) / "corpus")
            evaluador = EvaluadorFinanciero(corpus)
            respuesta = RespuestaAgente(
                respuesta="Las disrupciones pueden afectar las operaciones.",
                fuente="texto",
                citas=("ACME-2024-1A-0000",),
                cita="Esta frase no aparece en el informe.",
                latencia_ms=1,
            )

            cita_existe, cita_respalda = evaluador._evaluar_citas(respuesta)

            self.assertTrue(cita_existe)
            self.assertFalse(cita_respalda)

    def test_acepta_cita_literal_aunque_no_contenga_el_ancla_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal) / "corpus")
            evaluador = EvaluadorFinanciero(corpus)
            respuesta = RespuestaAgente(
                respuesta="Hubo disrupciones en la cadena de suministro.",
                fuente="texto",
                citas=("ACME-2024-1A-0000",),
                cita="Supply chain disruptions could",
                latencia_ms=1,
            )

            cita_existe, cita_respalda = evaluador._evaluar_citas(respuesta)

            self.assertTrue(cita_existe)
            self.assertTrue(cita_respalda)

    def test_tolerancia_relativa_acepta_redondeo_del_uno_por_ciento(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal) / "corpus")
            evaluador = EvaluadorFinanciero(
                corpus,
                tolerancia_absoluta=0,
                tolerancia_relativa=0.01,
            )
            caso = CasoGolden.model_validate(caso_numerico())

            redondeada = RespuestaAgente(
                respuesta="99.5 USD",
                cifra=99.5,
                unidad="USD",
                fuente="xbrl",
                latencia_ms=1,
            )
            incorrecta = redondeada.model_copy(
                update={"respuesta": "98 USD", "cifra": 98}
            )

            self.assertTrue(evaluador._cifra_correcta(caso, redondeada))
            self.assertFalse(evaluador._cifra_correcta(caso, incorrecta))

    def test_trayectoria_solo_exige_los_nombres_de_las_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "comparativa.jsonl"
            escribir_jsonl(ruta, [caso_comparativo()])
            evaluador = EvaluadorFinanciero(corpus)

            llamadas = tuple(
                LlamadaHerramienta(
                    id=f"xbrl-{ejercicio}",
                    nombre="get_xbrl_fact",
                    argumentos={
                        "ticker": "ACME",
                        "fiscal_year": ejercicio,
                        "concept": "NetIncomeLoss",
                    },
                    duracion_ms=0,
                    resultado=f"{valor} USD",
                )
                for ejercicio, valor in ((2023, 80), (2024, 100))
            ) + (
                LlamadaHerramienta(
                    id="search-2024",
                    nombre="search_filings",
                    argumentos={
                        "query": "operational disruption",
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "item": "1A",
                        "k": 5,
                    },
                    duracion_ms=0,
                    chunk_ids=("ACME-2024-1A-0000",),
                    resultado=TEXTO_2024,
                ),
            )
            respuesta = RespuestaAgente(
                respuesta=(
                    "Aumentó de 80 a 100 USD y hubo disrupciones operativas."
                ),
                cifra=100,
                unidad="USD",
                fuente="ambas",
                citas=("ACME-2024-1A-0000",),
                cita=ANCLA_2024,
                llamadas=llamadas,
                latencia_ms=1,
            )

            informe = evaluador.evaluar(
                nombre_agente="agente-comparativo",
                responder=lambda _: respuesta,
                ruta_jsonl=ruta,
            )

            self.assertTrue(informe.resultados[0].trayectoria_correcta)

            respuesta_con_argumentos_distintos = respuesta.model_copy(
                update={"llamadas": llamadas[1:]}
            )
            informe_argumentos_distintos = evaluador.evaluar(
                nombre_agente="agente-comparativo",
                responder=lambda _: respuesta_con_argumentos_distintos,
                ruta_jsonl=ruta,
            )
            self.assertTrue(
                informe_argumentos_distintos.resultados[0].trayectoria_correcta
            )

            respuesta_sin_xbrl = respuesta.model_copy(
                update={"llamadas": llamadas[-1:]}
            )
            informe_sin_xbrl = evaluador.evaluar(
                nombre_agente="agente-comparativo",
                responder=lambda _: respuesta_sin_xbrl,
                ruta_jsonl=ruta,
            )
            self.assertFalse(
                informe_sin_xbrl.resultados[0].trayectoria_correcta
            )

    def test_comparativa_solo_xbrl_no_exige_cita_ni_calcula_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "comparativa_xbrl.jsonl"
            datos_caso = caso_comparativo()
            datos_caso["herramienta_esperada"] = ["get_xbrl_fact"]
            escribir_jsonl(ruta, [datos_caso])
            evaluador = EvaluadorFinanciero(corpus)
            llamadas = tuple(
                LlamadaHerramienta(
                    id=f"xbrl-{ejercicio}",
                    nombre="get_xbrl_fact",
                    argumentos={
                        "ticker": "ACME",
                        "fiscal_year": ejercicio,
                        "concept": "NetIncomeLoss",
                    },
                    duracion_ms=0,
                    resultado=f"{valor} USD",
                )
                for ejercicio, valor in ((2023, 80), (2024, 100))
            )
            respuesta = RespuestaAgente(
                respuesta="El beneficio aumentó de 80 a 100 USD.",
                cifra=100,
                unidad="USD",
                fuente="xbrl",
                llamadas=llamadas,
                latencia_ms=1,
            )

            informe = evaluador.evaluar(
                nombre_agente="comparativa-xbrl",
                responder=lambda _: respuesta,
                ruta_jsonl=ruta,
            )

            resultado = informe.resultados[0]
            self.assertIsNone(resultado.cita_existe)
            self.assertIsNone(resultado.cita_respalda)
            self.assertIsNone(resultado.recall_at_k)
            self.assertTrue(resultado.acierto)
            self.assertIsNone(informe.recall_at_k_medio)
            self.assertEqual(informe.cobertura_recall, 0)
            self.assertEqual(resultado.observaciones, ())

    def test_comparativa_con_busqueda_sigue_exigiendo_cita(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "comparativa_hibrida.jsonl"
            escribir_jsonl(ruta, [caso_comparativo()])
            evaluador = EvaluadorFinanciero(corpus)
            llamadas = (
                LlamadaHerramienta(
                    id="xbrl-2024",
                    nombre="get_xbrl_fact",
                    argumentos={
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "concept": "NetIncomeLoss",
                    },
                    duracion_ms=0,
                    resultado="100 USD",
                ),
                LlamadaHerramienta(
                    id="search-2024",
                    nombre="search_filings",
                    argumentos={
                        "query": "operational disruption",
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "item": "1A",
                        "k": 5,
                    },
                    duracion_ms=0,
                    chunk_ids=("ACME-2024-1A-0000",),
                    resultado=TEXTO_2024,
                ),
            )
            respuesta = RespuestaAgente(
                respuesta="Aumentó de 80 a 100 USD por disrupciones.",
                cifra=100,
                unidad="USD",
                fuente="ambas",
                llamadas=llamadas,
                latencia_ms=1,
            )

            informe = evaluador.evaluar(
                nombre_agente="comparativa-hibrida",
                responder=lambda _: respuesta,
                ruta_jsonl=ruta,
            )

            resultado = informe.resultados[0]
            self.assertFalse(resultado.cita_existe)
            self.assertFalse(resultado.cita_respalda)
            self.assertEqual(resultado.recall_at_k, 1)
            self.assertFalse(resultado.acierto)
            self.assertIn("Falta una cita", resultado.observaciones[0])

    def test_evalua_cifra_cita_trayectoria_recall_y_agregados(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "evaluacion.jsonl"
            escribir_jsonl(ruta, [caso_extractivo(), caso_numerico()])
            evaluador = EvaluadorFinanciero(
                corpus,
                k_retrieval=1,
                tolerancia_absoluta=0.1,
                tolerancia_relativa=0,
                numero_esperado=2,
            )

            def responder(pregunta: str) -> RespuestaAgente:
                if "riesgo" in pregunta:
                    llamada = LlamadaHerramienta(
                        id="call-search",
                        nombre="search_filings",
                        argumentos={
                            "query": "risk",
                            "ticker": "ACME",
                            "fiscal_year": 2024,
                            "item": "1A",
                            "k": 1,
                        },
                        duracion_ms=2,
                        chunk_ids=("ACME-2024-1A-0000",),
                        resultado=TEXTO_2024,
                    )
                    return RespuestaAgente(
                        respuesta="Las disrupciones pueden dañar las operaciones.",
                        fuente="texto",
                        citas=("ACME-2024-1A-0000",),
                        cita=f"  {ANCLA_2024.upper()}  ",
                        llamadas=(llamada,),
                        latencia_ms=10,
                        coste_usd=0.01,
                    )
                llamada = LlamadaHerramienta(
                    id="call-xbrl",
                    nombre="get_xbrl_fact",
                    argumentos={
                        "ticker": "ACME",
                        "fiscal_year": 2024,
                        "concept": "NetIncomeLoss",
                    },
                    duracion_ms=1,
                    resultado="100 USD",
                )
                return RespuestaAgente(
                    respuesta="El beneficio fue 100 USD.",
                    cifra=100.05,
                    unidad="usd",
                    fuente="xbrl",
                    llamadas=(llamada,),
                    latencia_ms=30,
                    coste_usd=0.03,
                )

            informe = evaluador.evaluar(
                nombre_agente="equipo-a", responder=responder, ruta_jsonl=ruta
            )
            self.assertEqual(informe.aciertos_totales, 2)
            self.assertEqual(informe.recall_at_k_medio, 1)
            self.assertEqual(informe.latencia_media_ms, 20)
            self.assertEqual(informe.coste_medio_usd, 0.02)
            self.assertEqual(informe.llamadas_por_pregunta, 1)
            self.assertIsNone(informe.resultados[0].justificacion_cita)
            self.assertEqual(
                informe.metodo_soporte_citas,
                "coincidencia_literal_normalizada_120",
            )

    def test_argumentos_incorrectos_no_afectan_la_trayectoria(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta = raiz / "numerico.jsonl"
            escribir_jsonl(ruta, [caso_numerico()])
            evaluador = EvaluadorFinanciero(corpus)
            llamada = LlamadaHerramienta(
                id="mal",
                nombre="get_xbrl_fact",
                argumentos={
                    "ticker": "ACME",
                    "fiscal_year": 2024,
                    "concept": "Revenues",
                },
                duracion_ms=0,
                resultado="100 USD",
            )
            informe = evaluador.evaluar(
                nombre_agente="a",
                ruta_jsonl=ruta,
                responder=lambda _: RespuestaAgente(
                    respuesta="100 USD",
                    cifra=100,
                    unidad="USD",
                    fuente="xbrl",
                    llamadas=(llamada,),
                    latencia_ms=0,
                ),
            )
            self.assertTrue(informe.resultados[0].trayectoria_correcta)
            self.assertTrue(informe.resultados[0].acierto)

    def test_persiste_cada_caso_y_reanuda_solo_los_pendientes(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta_golden = raiz / "golden.jsonl"
            ruta_progreso = raiz / "progreso" / "parcial.json"
            primero = caso_numerico()
            segundo = caso_numerico()
            segundo.update(
                {
                    "id": "num-002",
                    "pregunta": "¿Cuál fue el beneficio neto de 2023?",
                    "fiscal_year": 2023,
                    "cifra_esperada": 80.0,
                    "respuesta_esperada": "80 USD",
                }
            )
            escribir_jsonl(ruta_golden, [primero, segundo])

            llamadas_primera: list[str] = []

            def responder_interrumpiendo(pregunta: str) -> RespuestaAgente:
                llamadas_primera.append(pregunta)
                if "2023" in pregunta:
                    raise KeyboardInterrupt
                return self._respuesta_numerica(2024, 100)

            evaluador = EvaluadorFinanciero(
                corpus, ruta_progreso=ruta_progreso
            )
            with self.assertRaises(KeyboardInterrupt):
                evaluador.evaluar(
                    nombre_agente="agente-v1",
                    responder=responder_interrumpiendo,
                    ruta_jsonl=ruta_golden,
                )
            self.assertTrue(ruta_progreso.is_file())

            llamadas_segunda: list[str] = []

            def responder_reanudando(pregunta: str) -> RespuestaAgente:
                llamadas_segunda.append(pregunta)
                return self._respuesta_numerica(2023, 80)

            informe = EvaluadorFinanciero(
                corpus, ruta_progreso=ruta_progreso
            ).evaluar(
                nombre_agente="agente-v1",
                responder=responder_reanudando,
                ruta_jsonl=ruta_golden,
            )
            self.assertEqual(len(llamadas_primera), 2)
            self.assertEqual(llamadas_segunda, [segundo["pregunta"]])
            self.assertEqual(informe.aciertos_totales, 2)

    def test_un_429_no_se_persiste_como_resultado_definitivo(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta_golden = raiz / "golden.jsonl"
            ruta_progreso = raiz / "progreso.json"
            escribir_jsonl(ruta_golden, [caso_numerico()])
            evaluador = EvaluadorFinanciero(
                corpus, ruta_progreso=ruta_progreso
            )

            with self.assertRaises(RateLimitAgotadoError):
                evaluador.evaluar(
                    nombre_agente="agente-v1",
                    ruta_jsonl=ruta_golden,
                    responder=lambda _: RespuestaAgente(
                        respuesta="",
                        fuente="ninguna",
                        latencia_ms=1,
                        error=(
                            "TooManyRequestsResponseError: Rate limit exceeded"
                        ),
                    ),
                )
            self.assertFalse(ruta_progreso.exists())

            informe = evaluador.evaluar(
                nombre_agente="agente-v1",
                ruta_jsonl=ruta_golden,
                responder=lambda _: self._respuesta_numerica(2024, 100),
            )
            self.assertEqual(informe.aciertos_totales, 1)
            self.assertTrue(ruta_progreso.is_file())

    def test_repara_un_bad_request_generico_guardado_en_progreso(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta_golden = raiz / "golden.jsonl"
            ruta_progreso = raiz / "progreso.json"
            escribir_jsonl(ruta_golden, [caso_numerico()])
            evaluador = EvaluadorFinanciero(
                corpus, ruta_progreso=ruta_progreso
            )
            evaluador.evaluar(
                nombre_agente="agente-v1",
                ruta_jsonl=ruta_golden,
                responder=lambda _: self._respuesta_numerica(2024, 100),
            )

            datos = json.loads(ruta_progreso.read_text(encoding="utf-8"))
            datos["resultados"][0]["respuesta_agente"]["error"] = (
                "BadRequestResponseError: Provider returned error"
            )
            ruta_progreso.write_text(
                json.dumps(datos, ensure_ascii=False), encoding="utf-8"
            )
            llamadas = 0

            def responder(_: str) -> RespuestaAgente:
                nonlocal llamadas
                llamadas += 1
                return self._respuesta_numerica(2024, 100)

            informe = evaluador.evaluar(
                nombre_agente="agente-v1",
                ruta_jsonl=ruta_golden,
                responder=responder,
            )
            self.assertEqual(llamadas, 1)
            self.assertEqual(informe.aciertos_totales, 1)
            self.assertIsNone(
                informe.resultados[0].respuesta_agente.error
            )

    def test_reintenta_un_graph_recursion_guardado_en_progreso(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus, _ = crear_corpus_temporal(raiz / "corpus")
            ruta_golden = raiz / "golden.jsonl"
            ruta_progreso = raiz / "progreso.json"
            escribir_jsonl(ruta_golden, [caso_numerico()])
            evaluador = EvaluadorFinanciero(
                corpus, ruta_progreso=ruta_progreso
            )
            evaluador.evaluar(
                nombre_agente="agente-v1",
                ruta_jsonl=ruta_golden,
                responder=lambda _: self._respuesta_numerica(2024, 100),
            )

            datos = json.loads(ruta_progreso.read_text(encoding="utf-8"))
            datos["resultados"][0]["respuesta_agente"]["error"] = (
                "GraphRecursionError: Recursion limit of 25 reached"
            )
            ruta_progreso.write_text(
                json.dumps(datos, ensure_ascii=False), encoding="utf-8"
            )
            llamadas = 0

            def responder(_: str) -> RespuestaAgente:
                nonlocal llamadas
                llamadas += 1
                return self._respuesta_numerica(2024, 100)

            informe = evaluador.evaluar(
                nombre_agente="agente-v1",
                ruta_jsonl=ruta_golden,
                responder=responder,
            )
            self.assertEqual(llamadas, 1)
            self.assertEqual(informe.aciertos_totales, 1)
            self.assertIsNone(informe.resultados[0].respuesta_agente.error)

    @staticmethod
    def _respuesta_numerica(anio: int, cifra: float) -> RespuestaAgente:
        llamada = LlamadaHerramienta(
            id=f"xbrl-{anio}",
            nombre="get_xbrl_fact",
            argumentos={
                "ticker": "ACME",
                "fiscal_year": anio,
                "concept": "NetIncomeLoss",
            },
            duracion_ms=0,
            resultado=f"{cifra} USD",
        )
        return RespuestaAgente(
            respuesta=f"El beneficio fue {cifra} USD.",
            cifra=cifra,
            unidad="USD",
            fuente="xbrl",
            llamadas=(llamada,),
            latencia_ms=1,
        )


if __name__ == "__main__":
    unittest.main()
