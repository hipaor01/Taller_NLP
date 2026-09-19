from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from taller_nlp import (
    ConfiguracionAgente,
    ConstructorAgente,
    RegistroTelemetriaAuxiliar,
    RespuestaAgente,
)

from tests.support import crear_corpus_temporal, crear_fabrica_prueba


class MotorControlado:
    construcciones: list[dict[str, object]] = []

    def __init__(
        self,
        configuracion,
        herramientas,
        corpus,
        *,
        middlewares=(),
        modelo=None,
        control_peticiones=None,
        checkpointer=None,
        telemetria_auxiliar=None,
    ) -> None:
        self.construcciones.append(
            {
                "configuracion": configuracion,
                "herramientas": herramientas,
                "corpus": corpus,
                "middlewares": middlewares,
                "modelo": modelo,
                "control_peticiones": control_peticiones,
                "checkpointer": checkpointer,
                "telemetria_auxiliar": telemetria_auxiliar,
            }
        )

    def responder(self, pregunta: str) -> RespuestaAgente:
        return RespuestaAgente(
            respuesta=f"Respuesta a: {pregunta}",
            fuente="ninguna",
            latencia_ms=0,
        )


class TestConstructorAgente(unittest.TestCase):
    def setUp(self) -> None:
        MotorControlado.construcciones.clear()

    def test_expone_el_motor_ensamblado(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))
            configuracion = ConfiguracionAgente(
                modelo="modelo-prueba",
                system_prompt="Responde con evidencia.",
            )
            constructor = ConstructorAgente(
                "equipo-a",
                configuracion,
                corpus,
                crear_fabrica_prueba(corpus),
                telemetria_auxiliar=RegistroTelemetriaAuxiliar(),
            )

            with patch("taller_nlp.assembly.MotorLangChain", MotorControlado):
                checkpointer = InMemorySaver()
                motor = constructor.construir_motor(checkpointer=checkpointer)

            self.assertIsInstance(motor, MotorControlado)
            self.assertEqual(len(MotorControlado.construcciones), 1)
            construccion = MotorControlado.construcciones[0]
            self.assertIs(construccion["configuracion"], configuracion)
            self.assertIs(construccion["corpus"], corpus)
            self.assertIs(
                construccion["control_peticiones"],
                constructor.control_peticiones,
            )
            self.assertIs(construccion["checkpointer"], checkpointer)
            self.assertIs(
                construccion["telemetria_auxiliar"],
                constructor.telemetria_auxiliar,
            )

    def test_ensambla_fachada_con_componentes_del_mismo_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))
            configuracion = ConfiguracionAgente(
                modelo="modelo-prueba",
                system_prompt="Responde con evidencia.",
            )
            constructor = ConstructorAgente(
                " equipo-a ",
                configuracion,
                corpus,
                crear_fabrica_prueba(corpus),
                k_retrieval=3,
            )

            with patch("taller_nlp.assembly.MotorLangChain", MotorControlado):
                agente = constructor.construir()

            self.assertEqual(agente.nombre, "equipo-a")
            self.assertEqual(
                agente.responder("pregunta").respuesta,
                "Respuesta a: pregunta",
            )
            construccion = MotorControlado.construcciones[0]
            self.assertIs(construccion["configuracion"], configuracion)
            self.assertIs(construccion["corpus"], corpus)
            self.assertIs(constructor.evaluador.corpus, corpus)
            self.assertEqual(constructor.evaluador.k_retrieval, 3)
            self.assertEqual(
                [tool.name for tool in construccion["herramientas"].herramientas],
                [
                    "list_available",
                    "get_xbrl_fact",
                    "search_filings",
                    "read_section",
                ],
            )

    def test_rechaza_fabrica_ligada_a_otra_variante(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            corpus_a, _ = crear_corpus_temporal(raiz / "a", nombre="a")
            corpus_b, _ = crear_corpus_temporal(raiz / "b", nombre="b")
            configuracion = ConfiguracionAgente(
                modelo="modelo", system_prompt="prompt"
            )
            with self.assertRaisesRegex(ValueError, "misma CorpusVariant"):
                ConstructorAgente(
                    "agente",
                    configuracion,
                    corpus_a,
                    crear_fabrica_prueba(corpus_b),
                )

    def test_reutiliza_validacion_del_evaluador(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, _ = crear_corpus_temporal(Path(temporal))
            configuracion = ConfiguracionAgente(
                modelo="modelo", system_prompt="prompt"
            )
            with self.assertRaisesRegex(ValueError, "k_retrieval"):
                ConstructorAgente(
                    "agente",
                    configuracion,
                    corpus,
                    crear_fabrica_prueba(corpus),
                    k_retrieval=0,
                )


if __name__ == "__main__":
    unittest.main()
