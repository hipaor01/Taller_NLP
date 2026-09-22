from __future__ import annotations

import tempfile
import unittest
import os
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage

import miax_s2
from agente.interfaz import (
    MODULO_AGENTE_PREDETERMINADO,
    VARIABLE_VARIANTE,
    _RuntimeNotebook,
    _crear_constructor_configurado,
    resumir,
)
from taller_nlp import (
    CasoGolden,
    EjecucionLangChain,
    InformeEvaluacion,
    LlamadaHerramienta,
    LlamadaModeloAuxiliar,
    RespuestaAgente,
    RespuestaFinanciera,
    ResultadoPregunta,
)


class _MotorControlado:
    def __init__(self, ejecucion: EjecucionLangChain) -> None:
        self.ejecucion = ejecucion
        self.invocaciones: list[tuple[str, str | None]] = []

    def ejecutar(
        self,
        pregunta: str,
        *,
        thread_id: str | None = None,
    ) -> EjecucionLangChain:
        self.invocaciones.append((pregunta, thread_id))
        return self.ejecucion

    def responder(self, pregunta: str) -> object:
        return object()


class TestInterfazNotebook(unittest.TestCase):
    def test_resumir_genera_fila_comparativa(self) -> None:
        import pandas as pd

        tabla = pd.DataFrame(
            {
                "familia": [
                    "extractiva",
                    "numerica",
                    "comparativa",
                    "comparativa",
                ],
                "acierto": [True, True, True, False],
                "cita": [True, None, True, False],
                "cifra": [None, True, True, True],
                "trayectoria": [True, True, True, False],
                "recall": [1.0, None, 1.0, 0.0],
                "coste_usd": [0.01, 0.03, 0.02, 0.02],
                "latencia_s": [1.0, 3.0, 2.0, 2.0],
                "llamadas": [2, 4, 3, 3],
            }
        )

        resumen = resumir(tabla, "  baseline  ")

        self.assertEqual(resumen["versión"], "baseline")
        self.assertEqual(resumen["cita"], 2 / 3)
        self.assertEqual(resumen["cifra"], 1.0)
        self.assertEqual(resumen["trayectoria"], 0.75)
        self.assertEqual(resumen["recall@5"], 2 / 3)
        self.assertEqual(resumen["coste medio (¢)"], 2.0)
        self.assertEqual(resumen["latencia media (s)"], 2.0)
        self.assertEqual(resumen["llamadas/pregunta"], 3.0)
        self.assertEqual(resumen["aciertos extractiva"], "1/1")
        self.assertEqual(resumen["aciertos numerica"], "1/1")
        self.assertEqual(resumen["aciertos comparativa"], "1/2")

    def test_resumir_reconstruye_aciertos_de_un_csv_antiguo(self) -> None:
        import pandas as pd

        tabla = pd.DataFrame(
            {
                "familia": ["extractiva", "numerica", "comparativa"],
                "cita": [True, None, True],
                "cifra": [None, True, True],
                "trayectoria": [True, False, True],
            }
        )

        resumen = resumir(tabla, "anterior")

        self.assertEqual(resumen["aciertos extractiva"], "1/1")
        self.assertEqual(resumen["aciertos numerica"], "0/1")
        self.assertEqual(resumen["aciertos comparativa"], "1/1")

    def test_resumir_valida_entrada_y_admite_columnas_ausentes(self) -> None:
        import pandas as pd

        with self.assertRaisesRegex(TypeError, "DataFrame"):
            resumir([], "baseline")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "etiqueta"):
            resumir(pd.DataFrame(), "  ")

        resumen = resumir(pd.DataFrame(), "vacío")
        self.assertTrue(math.isnan(resumen["cita"]))
        self.assertTrue(math.isnan(resumen["coste medio (¢)"]))

    def test_selecciona_jchulvi_v2_por_defecto(self) -> None:
        constructor = object()
        modulo = SimpleNamespace(crear_constructor=lambda: constructor)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(VARIABLE_VARIANTE, None)
            with patch("agente.interfaz.import_module", return_value=modulo) as cargar:
                resultado = _crear_constructor_configurado()

        self.assertIs(resultado, constructor)
        cargar.assert_called_once_with(MODULO_AGENTE_PREDETERMINADO)
        self.assertEqual(
            MODULO_AGENTE_PREDETERMINADO,
            "experimentos.jchulvi.agente_v2",
        )

    def test_selecciona_otra_variante_por_variable_de_entorno(self) -> None:
        constructor = object()
        modulo = SimpleNamespace(crear_constructor=lambda: constructor)
        nombre = "experimentos.higinio.agente_v002"
        with patch.dict(os.environ, {VARIABLE_VARIANTE: nombre}):
            with patch("agente.interfaz.import_module", return_value=modulo) as cargar:
                resultado = _crear_constructor_configurado()

        self.assertIs(resultado, constructor)
        cargar.assert_called_once_with(nombre)

    def test_rechaza_variante_sin_factoria_comun(self) -> None:
        with patch.dict(
            os.environ,
            {VARIABLE_VARIANTE: "experimentos.incompleto"},
        ):
            with patch(
                "agente.interfaz.import_module",
                return_value=SimpleNamespace(),
            ):
                with self.assertRaisesRegex(AttributeError, "crear_constructor"):
                    _crear_constructor_configurado()

    def test_responder_conserva_estado_telemetria_y_thread_id(self) -> None:
        mensajes = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_filings",
                        "args": {"query": "revenue"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="[AAPL-2025-7-0001] evidencia",
                tool_call_id="call-1",
                name="search_filings",
            ),
        ]
        estructurada = RespuestaFinanciera(
            respuesta="Respuesta de prueba.",
            fuente="texto",
            cita="evidencia",
            chunk_id="AAPL-2025-7-0001",
        )
        llamada_auxiliar = LlamadaModeloAuxiliar(
            componente="reescritor_consulta",
            modelo="modelo-auxiliar",
            latencia_ms=25,
            tokens_entrada=4,
            tokens_salida=2,
            coste_usd=0.0001,
        )
        motor = _MotorControlado(
            EjecucionLangChain(
                estado={
                    "messages": mensajes,
                    "structured_response": estructurada,
                },
                latencia_ms=1_250,
                coste_usd=0.002,
                tokens_entrada=10,
                tokens_salida=5,
                llamadas_modelo_auxiliares=(llamada_auxiliar,),
                coste_modelo_principal_usd=0.0019,
                tokens_entrada_modelo_principal=6,
                tokens_salida_modelo_principal=3,
            )
        )
        runtime = _RuntimeNotebook(
            constructor=Mock(sesion_ejecucion=nullcontext),
            motor=motor,  # type: ignore[arg-type]
        )

        resultado = runtime.responder("  Pregunta  ", thread_id="hilo-1")

        self.assertIs(resultado["messages"], mensajes)
        self.assertIs(resultado["structured_response"], estructurada)
        self.assertEqual(resultado["coste_usd"], 0.002)
        self.assertEqual(resultado["latencia_s"], 1.25)
        self.assertEqual(resultado["tokens_entrada"], 10)
        self.assertEqual(resultado["tokens_salida"], 5)
        self.assertEqual(
            resultado["llamadas_modelo_auxiliares"],
            (llamada_auxiliar,),
        )
        self.assertEqual(resultado["coste_modelo_principal_usd"], 0.0019)
        self.assertEqual(resultado["tokens_entrada_modelo_principal"], 6)
        self.assertEqual(resultado["tokens_salida_modelo_principal"], 3)
        self.assertEqual(motor.invocaciones, [("Pregunta", "hilo-1")])
        self.assertEqual(
            miax_s2.herramientas_usadas(resultado),
            ["search_filings"],
        )

    def test_responder_aplica_valores_compatibles_con_el_notebook(self) -> None:
        motor = _MotorControlado(
            EjecucionLangChain(
                estado={"messages": (), "structured_response": None},
                latencia_ms=0,
                coste_usd=None,
                tokens_entrada=None,
                tokens_salida=None,
            )
        )
        runtime = _RuntimeNotebook(
            constructor=Mock(sesion_ejecucion=nullcontext),
            motor=motor,  # type: ignore[arg-type]
        )

        resultado = runtime.responder("Pregunta")

        self.assertEqual(resultado["coste_usd"], 0.0)
        self.assertEqual(resultado["latencia_s"], 0.0)
        self.assertEqual(motor.invocaciones, [("Pregunta", "s2")])

    def test_evaluar_informe_delega_en_el_evaluador_del_constructor(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "holdout.jsonl"
            ruta.write_text("{}\n", encoding="utf-8")
            informe = object()
            evaluador = Mock()
            evaluador.evaluar.return_value = informe
            constructor = SimpleNamespace(
                nombre="baseline-notebook-s1",
                evaluador=evaluador,
                sesion_ejecucion=nullcontext,
            )
            motor = _MotorControlado(
                EjecucionLangChain(
                    estado={},
                    latencia_ms=0,
                    coste_usd=None,
                    tokens_entrada=None,
                    tokens_salida=None,
                )
            )
            runtime = _RuntimeNotebook(
                constructor=constructor,  # type: ignore[arg-type]
                motor=motor,  # type: ignore[arg-type]
            )

            resultado = runtime.evaluar_informe(ruta)

            self.assertIs(resultado, informe)
            evaluador.evaluar.assert_called_once_with(
                nombre_agente="baseline-notebook-s1",
                responder=motor.responder,
                ruta_jsonl=ruta,
            )

    def test_evaluar_devuelve_dataframe_y_guarda_csv_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            ruta = raiz / "holdout.jsonl"
            ruta.write_text("{}\n", encoding="utf-8")
            salida = raiz / "resultados" / "resultado.csv"
            caso = CasoGolden.model_validate(
                {
                    "id": "num-001",
                    "pregunta": "¿Cuál fue el beneficio?",
                    "familia": "numerica",
                    "ticker": "ACME",
                    "fiscal_year": 2024,
                    "respuesta_esperada": "100 USD",
                    "cifra_esperada": 100,
                    "unidad": "USD",
                    "concept_xbrl": "NetIncomeLoss",
                    "herramienta_esperada": ["get_xbrl_fact"],
                    "autor": "equipo",
                }
            )
            llamada = LlamadaHerramienta(
                id="xbrl-1",
                nombre="get_xbrl_fact",
                argumentos={
                    "ticker": "ACME",
                    "fiscal_year": 2024,
                    "concept": "NetIncomeLoss",
                },
                duracion_ms=1,
                resultado="100 USD",
            )
            respuesta = RespuestaAgente(
                respuesta="100 USD",
                cifra=100,
                unidad="USD",
                fuente="xbrl",
                llamadas=(llamada,),
                latencia_ms=1_500,
                coste_usd=None,
            )
            informe = InformeEvaluacion(
                nombre_agente="baseline-notebook-s1",
                ruta_jsonl=str(ruta),
                resultados=(
                    ResultadoPregunta(
                        id_pregunta="num-001",
                        familia="numerica",
                        respuesta_agente=respuesta,
                        cifra_correcta=True,
                        trayectoria_correcta=True,
                    ),
                ),
                k_retrieval=5,
                tolerancia_absoluta=0,
                tolerancia_relativa=0.01,
            )
            evaluador = Mock(
                numero_esperado=None,
                minimo_comparativas=0,
            )
            evaluador.evaluar.return_value = informe
            constructor = SimpleNamespace(
                nombre="baseline-notebook-s1",
                evaluador=evaluador,
                corpus=object(),
                sesion_ejecucion=nullcontext,
            )
            runtime = _RuntimeNotebook(
                constructor=constructor,  # type: ignore[arg-type]
                motor=_MotorControlado(
                    EjecucionLangChain(
                        estado={},
                        latencia_ms=0,
                        coste_usd=None,
                        tokens_entrada=None,
                        tokens_salida=None,
                    )
                ),  # type: ignore[arg-type]
            )

            with patch(
                "taller_nlp.CasoGolden.cargar_jsonl",
                return_value=(caso,),
            ):
                tabla = runtime.evaluar(ruta, salida=salida)

            self.assertEqual(
                list(tabla.columns),
                [
                    "id",
                    "familia",
                    "ticker",
                    "acierto",
                    "latencia_s",
                    "coste_usd",
                    "llamadas",
                    "cita",
                    "cifra",
                    "trayectoria",
                    "recall",
                    "error",
                ],
            )
            self.assertEqual(tabla.loc[0, "ticker"], "ACME")
            self.assertEqual(tabla.loc[0, "latencia_s"], 1.5)
            self.assertEqual(tabla.loc[0, "coste_usd"], 0.0)
            self.assertEqual(tabla.loc[0, "llamadas"], 1)
            self.assertTrue(tabla.loc[0, "cifra"])
            self.assertTrue(tabla.loc[0, "trayectoria"])
            self.assertTrue(salida.is_file())


if __name__ == "__main__":
    unittest.main()
