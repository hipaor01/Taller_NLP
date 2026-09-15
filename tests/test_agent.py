from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from taller_nlp import AgenteFinanciero, InformeEvaluacion, RespuestaAgente


class MotorControlado:
    def __init__(self, salida: object) -> None:
        self.salida = salida
        self.preguntas: list[str] = []

    def responder(self, pregunta: str) -> object:
        self.preguntas.append(pregunta)
        if isinstance(self.salida, Exception):
            raise self.salida
        return self.salida


class EvaluadorControlado:
    def __init__(self, informe: InformeEvaluacion) -> None:
        self.informe = informe
        self.nombre_recibido: str | None = None

    def evaluar(self, *, nombre_agente, responder, ruta_jsonl):
        del responder, ruta_jsonl
        self.nombre_recibido = nombre_agente
        return self.informe


def informe_vacio_logico(nombre: str) -> InformeEvaluacion:
    from taller_nlp import ResultadoPregunta

    respuesta = RespuestaAgente(
        respuesta="ok", fuente="ninguna", latencia_ms=0
    )
    resultado = ResultadoPregunta(
        id_pregunta="q1",
        familia="numerica",
        respuesta_agente=respuesta,
        cifra_correcta=False,
        trayectoria_correcta=False,
    )
    return InformeEvaluacion(
        nombre_agente=nombre,
        ruta_jsonl="golden.jsonl",
        resultados=(resultado,),
        k_retrieval=1,
        tolerancia_absoluta=0,
        tolerancia_relativa=0,
    )


class TestAgenteFinanciero(unittest.TestCase):
    def test_responder_normaliza_pregunta_y_mide_latencia_en_fachada(self) -> None:
        motor = MotorControlado(
            RespuestaAgente(respuesta="ok", fuente="ninguna", latencia_ms=999)
        )
        agente = AgenteFinanciero(
            "equipo-a", motor, EvaluadorControlado(informe_vacio_logico("equipo-a"))
        )
        salida = agente.responder("  pregunta  ")
        self.assertEqual(motor.preguntas, ["pregunta"])
        self.assertGreaterEqual(salida.latencia_ms, 0)
        self.assertNotEqual(salida.latencia_ms, 999)

    def test_responder_convierte_excepcion_del_motor_en_respuesta(self) -> None:
        agente = AgenteFinanciero(
            "equipo-a",
            MotorControlado(RuntimeError("fallo controlado")),
            EvaluadorControlado(informe_vacio_logico("equipo-a")),
        )
        salida = agente.responder("pregunta")
        self.assertFalse(salida.exitosa)
        self.assertIn("RuntimeError: fallo controlado", salida.error or "")

    def test_rechaza_tipo_de_salida_incorrecto_del_motor(self) -> None:
        agente = AgenteFinanciero(
            "equipo-a",
            MotorControlado("texto suelto"),
            EvaluadorControlado(informe_vacio_logico("equipo-a")),
        )
        with self.assertRaisesRegex(TypeError, "RespuestaAgente"):
            agente.responder("pregunta")

    def test_evaluar_delega_y_exige_mismo_nombre(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "ciego.jsonl"
            ruta.write_text("{}\n", encoding="utf-8")
            evaluador = EvaluadorControlado(informe_vacio_logico("equipo-a"))
            agente = AgenteFinanciero(
                "equipo-a",
                MotorControlado(
                    RespuestaAgente(
                        respuesta="ok", fuente="ninguna", latencia_ms=0
                    )
                ),
                evaluador,
            )
            self.assertEqual(agente.evaluar(ruta).nombre_agente, "equipo-a")
            self.assertEqual(evaluador.nombre_recibido, "equipo-a")


if __name__ == "__main__":
    unittest.main()
