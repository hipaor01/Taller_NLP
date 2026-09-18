from __future__ import annotations

import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from taller_nlp import (
    ConfiguracionAgente,
    LlamadaModeloAuxiliar,
    MotorLangChain,
    RegistroTelemetriaAuxiliar,
    RespuestaFinanciera,
)
from taller_nlp import ControlPeticionesModelo


class _AgenteControlado:
    def __init__(self, al_invocar=None) -> None:
        self.configuraciones: list[dict] = []
        self.al_invocar = al_invocar
        self.estado = {
            "messages": (),
            "structured_response": RespuestaFinanciera(
                respuesta="Respuesta de prueba.",
                fuente="ninguna",
            ),
        }

    def invoke(self, entrada: dict, config: dict) -> dict:
        self.configuraciones.append(config)
        if self.al_invocar is not None:
            self.al_invocar()
        return self.estado


class TestTrazaLangChain(unittest.TestCase):
    def test_ejecutar_expone_el_estado_crudo_y_su_telemetria(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        agente = _AgenteControlado()
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = None

        ejecucion = motor.ejecutar("Pregunta de prueba")

        self.assertIs(ejecucion.estado, agente.estado)
        self.assertGreaterEqual(ejecucion.latencia_ms, 0)
        self.assertEqual(ejecucion.latencia_s, ejecucion.latencia_ms / 1_000)
        self.assertIsNone(ejecucion.coste_usd)
        self.assertIsNone(ejecucion.tokens_entrada)
        self.assertIsNone(ejecucion.tokens_salida)

    def test_reserva_supersteps_suficientes_para_las_iteraciones(self) -> None:
        for max_iteraciones, esperado in ((3, 50), (8, 64)):
            with self.subTest(max_iteraciones=max_iteraciones):
                configuracion = ConfiguracionAgente(
                    modelo="modelo",
                    system_prompt="prompt",
                    max_iteraciones=max_iteraciones,
                )
                agente = _AgenteControlado()
                motor = object.__new__(MotorLangChain)
                motor._configuracion = configuracion
                motor._corpus = SimpleNamespace(nombre="prueba")
                motor._agente = agente
                motor._checkpointer = None

                motor.responder("Pregunta de prueba")

                self.assertEqual(
                    agente.configuraciones[0]["recursion_limit"], esperado
                )

    def test_thread_id_configura_el_hilo_y_se_normaliza(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        agente = _AgenteControlado()
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = object()

        motor.ejecutar("Pregunta", thread_id=" conversacion-1 ")

        self.assertEqual(
            agente.configuraciones[0]["configurable"],
            {"thread_id": "conversacion-1"},
        )

    def test_sin_thread_id_usa_hilos_efimeros_independientes(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        agente = _AgenteControlado()
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = object()

        motor.ejecutar("Primera")
        motor.ejecutar("Segunda")

        hilos = [
            configuracion["configurable"]["thread_id"]
            for configuracion in agente.configuraciones
        ]
        self.assertTrue(all(hilos))
        self.assertNotEqual(hilos[0], hilos[1])

    def test_rechaza_thread_id_invalido(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = _AgenteControlado()
        motor._checkpointer = None

        with self.assertRaisesRegex(TypeError, "thread_id"):
            motor.ejecutar("Pregunta", thread_id=123)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "thread_id"):
            motor.ejecutar("Pregunta", thread_id="  ")

    def test_materializa_limites_globales_y_por_herramienta(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
            max_iteraciones=3,
            max_llamadas_total=4,
            limites_por_herramienta={
                "read_section": 0,
                "search_filings": 2,
            },
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._middlewares_usuario = ()
        motor._control_peticiones = ControlPeticionesModelo.desde_configuracion(
            configuracion
        )

        middlewares = motor._crear_middlewares()
        limites = [
            (
                type(middleware).__name__,
                getattr(middleware, "tool_name", None),
                getattr(middleware, "run_limit", None),
            )
            for middleware in middlewares
        ]
        self.assertIn(("ModelCallLimitMiddleware", None, 3), limites)
        self.assertIn(("ToolCallLimitMiddleware", None, 4), limites)
        self.assertIn(("ToolCallLimitMiddleware", "read_section", 0), limites)
        self.assertIn(("ToolCallLimitMiddleware", "search_filings", 2), limites)

    def test_extrae_llamadas_chunk_ids_duracion_y_errores(self) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_filings",
                    "args": {"query": "risk"},
                    "id": "call-1",
                    "type": "tool_call",
                },
                {
                    "name": "get_xbrl_fact",
                    "args": {"ticker": "ACME"},
                    "id": "call-2",
                    "type": "tool_call",
                },
            ],
        )
        resultado = ToolMessage(
            content="[ACME-2024-1A-0000] ACME FY2024 Item 1A\ntexto",
            tool_call_id="call-1",
            name="search_filings",
            additional_kwargs={"taller_nlp_duracion_ms": 12.5},
        )
        error = ToolMessage(
            content="fallo de XBRL",
            tool_call_id="call-2",
            name="get_xbrl_fact",
            status="error",
        )
        llamadas = MotorLangChain._extraer_llamadas((ai, resultado, error))
        self.assertEqual(llamadas[0].chunk_ids, ("ACME-2024-1A-0000",))
        self.assertEqual(llamadas[0].duracion_ms, 12.5)
        self.assertTrue(llamadas[0].exitosa)
        self.assertEqual(llamadas[1].error, "fallo de XBRL")
        self.assertFalse(llamadas[1].exitosa)

    def test_registra_tool_call_sin_resultado_como_error(self) -> None:
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_section",
                    "args": {"ticker": "ACME", "fiscal_year": 2024, "item": "1A"},
                    "id": "call-ausente",
                    "type": "tool_call",
                }
            ],
        )
        llamada = MotorLangChain._extraer_llamadas((ai,))[0]
        self.assertIn("no produjo un ToolMessage", llamada.error or "")

    def test_responder_conserva_cita_textual_y_chunk_id(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        agente = _AgenteControlado()
        agente.estado["structured_response"] = RespuestaFinanciera(
            respuesta="Respuesta con evidencia.",
            fuente="texto",
            cita="Texto literal del informe.",
            chunk_id="ACME-2024-1A-0000",
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = None

        respuesta = motor.responder("Pregunta")

        self.assertEqual(respuesta.cita, "Texto literal del informe.")
        self.assertEqual(respuesta.citas, ("ACME-2024-1A-0000",))

    def test_suma_tokens_y_coste_de_multiples_mensajes(self) -> None:
        mensajes = (
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                response_metadata={"cost": 0.01},
            ),
            AIMessage(
                content="b",
                usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                response_metadata={"cost": 0.02},
            ),
        )
        self.assertEqual(MotorLangChain._extraer_uso(mensajes), (15, 5, 0.03))

    def test_suma_telemetria_auxiliar_y_conserva_desglose(self) -> None:
        registro = RegistroTelemetriaAuxiliar()
        llamada_auxiliar = LlamadaModeloAuxiliar(
            componente="reescritor_consulta",
            modelo="modelo-auxiliar",
            latencia_ms=4,
            tokens_entrada=10,
            tokens_salida=5,
            coste_usd=0.002,
        )
        agente = _AgenteControlado(
            al_invocar=lambda: registro.registrar(llamada_auxiliar)
        )
        agente.estado["messages"] = (
            AIMessage(
                content="respuesta",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
                response_metadata={"cost": 0.01},
            ),
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
        )
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = None
        motor._telemetria_auxiliar = registro

        ejecucion = motor.ejecutar("Pregunta")

        self.assertEqual(ejecucion.tokens_entrada, 110)
        self.assertEqual(ejecucion.tokens_salida, 25)
        self.assertAlmostEqual(ejecucion.coste_usd or 0, 0.012)
        self.assertEqual(ejecucion.tokens_entrada_modelo_principal, 100)
        self.assertEqual(ejecucion.tokens_salida_modelo_principal, 20)
        self.assertEqual(ejecucion.coste_modelo_principal_usd, 0.01)
        self.assertEqual(
            ejecucion.llamadas_modelo_auxiliares,
            (llamada_auxiliar,),
        )

    def test_total_es_desconocido_si_falta_telemetria_auxiliar(self) -> None:
        llamada = LlamadaModeloAuxiliar(
            componente="reescritor_consulta",
            modelo="modelo-auxiliar",
            latencia_ms=1,
        )

        self.assertIsNone(
            MotorLangChain._sumar_telemetria(0.01, (llamada,), "coste_usd")
        )

    def test_estima_coste_si_el_proveedor_no_lo_informa(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
            precio_entrada_usd_millon_tokens=0.75,
            precio_salida_usd_millon_tokens=3.75,
        )
        agente = _AgenteControlado()
        agente.estado["messages"] = (
            AIMessage(
                content="respuesta",
                usage_metadata={
                    "input_tokens": 1_000,
                    "output_tokens": 200,
                    "total_tokens": 1_200,
                },
            ),
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = None

        ejecucion = motor.ejecutar("Pregunta")

        self.assertAlmostEqual(ejecucion.coste_usd or 0, 0.0015)

    def test_prioriza_coste_informado_sobre_la_estimacion(self) -> None:
        configuracion = ConfiguracionAgente(
            modelo="modelo",
            system_prompt="prompt",
            precio_entrada_usd_millon_tokens=100,
            precio_salida_usd_millon_tokens=100,
        )
        agente = _AgenteControlado()
        agente.estado["messages"] = (
            AIMessage(
                content="respuesta",
                usage_metadata={
                    "input_tokens": 1_000,
                    "output_tokens": 200,
                    "total_tokens": 1_200,
                },
                response_metadata={"cost": 0.0123},
            ),
        )
        motor = object.__new__(MotorLangChain)
        motor._configuracion = configuracion
        motor._corpus = SimpleNamespace(nombre="prueba")
        motor._agente = agente
        motor._checkpointer = None

        ejecucion = motor.ejecutar("Pregunta")

        self.assertEqual(ejecucion.coste_usd, 0.0123)


if __name__ == "__main__":
    unittest.main()
