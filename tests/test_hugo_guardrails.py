"""Pruebas de los guardarraíles de Hugo: cifras XBRL y reparación de citas."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from experimentos.hugo.guardrails import (
    MARCA_CIFRAS,
    ComprobadorCifrasXBRL,
    ReparadorCitas,
    buscar_literal,
    chunks_recuperados,
    volcar_intervenciones,
)
from taller_nlp import RespuestaFinanciera
from tests.support import TEXTO_2024, crear_corpus_temporal


def _respuesta(**cambios: object) -> RespuestaFinanciera:
    campos: dict[str, object] = {
        "respuesta": "Respuesta de prueba.",
        "fuente": "xbrl",
        "ticker": "ACME",
        "ejercicio": 2024,
    }
    campos.update(cambios)
    return RespuestaFinanciera(**campos)


def _llamada_xbrl(
    ticker: str = "ACME",
    fiscal_year: int = 2024,
    concept: str = "NetIncomeLoss",
    *,
    identificador: str = "call-xbrl",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_xbrl_fact",
                "args": {
                    "ticker": ticker,
                    "fiscal_year": fiscal_year,
                    "concept": concept,
                },
                "id": identificador,
            }
        ],
    )


def _busqueda(chunk_id: str, texto: str) -> list[object]:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_filings",
                    "args": {"query": "risks"},
                    "id": "call-busqueda",
                }
            ],
        ),
        ToolMessage(
            content=f"[{chunk_id}] ACME FY2024 Item 1A\n{texto}",
            tool_call_id="call-busqueda",
            name="search_filings",
        ),
    ]


class TestComprobadorCifrasXBRL(unittest.TestCase):
    """El corpus de prueba tiene NetIncomeLoss = 100 (2024) y 80 (2023)."""

    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.corpus, self.fragmentos = crear_corpus_temporal(
            Path(self.temporal.name)
        )
        self.comprobador = ComprobadorCifrasXBRL(self.corpus)
        self.runtime = Runtime()

    def tearDown(self) -> None:
        self.temporal.cleanup()

    def _ejecutar(
        self,
        respuesta: RespuestaFinanciera,
        mensajes: list[object] | None = None,
    ) -> dict | None:
        return self.comprobador.after_model(
            {
                "messages": mensajes if mensajes is not None else [],
                "structured_response": respuesta,
            },
            self.runtime,
        )

    def test_calla_si_no_hay_cifra_o_no_hay_respuesta_estructurada(self) -> None:
        self.assertIsNone(self._ejecutar(_respuesta(cifra=None)))
        self.assertIsNone(
            self.comprobador.after_model(
                {"messages": [], "structured_response": None}, self.runtime
            )
        )

    def test_acepta_la_cifra_consultada_con_get_xbrl_fact(self) -> None:
        resultado = self._ejecutar(
            _respuesta(cifra=100.0, unidad="USD"), [_llamada_xbrl()]
        )
        self.assertIsNone(resultado)

    def test_normaliza_la_unidad_sin_llamar_al_modelo(self) -> None:
        resultado = self._ejecutar(
            _respuesta(cifra=100.0, unidad="dólares"), [_llamada_xbrl()]
        )
        assert resultado is not None
        self.assertNotIn("jump_to", resultado)
        self.assertEqual(resultado["structured_response"].unidad, "USD")
        self.assertEqual(resultado["structured_response"].cifra, 100.0)

    def test_reclama_la_cifra_leida_del_texto(self) -> None:
        """La cifra existe en el XBRL, pero no se pidió con get_xbrl_fact."""
        resultado = self._ejecutar(_respuesta(cifra=100.0, unidad="USD"))
        assert resultado is not None
        self.assertEqual(resultado["jump_to"], "model")
        mensaje = resultado["messages"][0]["content"]
        self.assertIn(MARCA_CIFRAS, mensaje)
        self.assertIn("get_xbrl_fact", mensaje)
        self.assertIn("NetIncomeLoss", mensaje)

    def test_reclama_la_cifra_del_ejercicio_equivocado(self) -> None:
        resultado = self._ejecutar(
            _respuesta(cifra=80.0, unidad="USD", ejercicio=2024),
            [_llamada_xbrl(fiscal_year=2023)],
        )
        assert resultado is not None
        mensaje = resultado["messages"][0]["content"]
        self.assertIn("FY2023", mensaje)
        self.assertIn("MÁS RECIENTE", mensaje)

    def test_reclama_una_cifra_que_no_esta_en_el_xbrl(self) -> None:
        resultado = self._ejecutar(
            _respuesta(cifra=250.0, unidad="USD"), [_llamada_xbrl()]
        )
        assert resultado is not None
        mensaje = resultado["messages"][0]["content"]
        self.assertIn("Has consultado: ACME FY2024 NetIncomeLoss = 100 USD", mensaje)
        self.assertIn("Conceptos de ACME FY2024", mensaje)

    def test_da_el_valor_exacto_si_la_cifra_es_el_redondeo_de_la_herramienta(
        self,
    ) -> None:
        """get_xbrl_fact imprime 7,46 USD/shares como «7»."""
        import pandas as pd

        tabla = pd.read_parquet(self.corpus.ruta_xbrl)
        fila = tabla.iloc[0].to_dict() | {
            "concept": "EarningsPerShareDiluted",
            "value": 7.46,
            "unit": "USD/shares",
        }
        pd.concat([tabla, pd.DataFrame([fila])]).to_parquet(
            self.corpus.ruta_xbrl, index=False
        )
        comprobador = ComprobadorCifrasXBRL(self.corpus)
        resultado = comprobador.after_model(
            {
                "messages": [
                    _llamada_xbrl(concept="EarningsPerShareDiluted")
                ],
                "structured_response": _respuesta(
                    cifra=7.0, unidad="USD/shares"
                ),
            },
            self.runtime,
        )
        assert resultado is not None
        mensaje = resultado["messages"][0]["content"]
        self.assertIn("redondeado", mensaje)
        self.assertIn("cifra=7.46", mensaje)

    def test_recuerda_responder_ninguna_si_la_compania_no_esta(self) -> None:
        resultado = self._ejecutar(
            _respuesta(cifra=500.0, unidad="USD", ticker="TSLA", ejercicio=2025)
        )
        assert resultado is not None
        mensaje = resultado["messages"][0]["content"]
        self.assertIn("fuente='ninguna'", mensaje)
        self.assertIn("ACME", mensaje)

    def test_acepta_una_cifra_derivada_de_los_hechos_consultados(self) -> None:
        mensajes = [
            _llamada_xbrl(fiscal_year=2024, identificador="a"),
            _llamada_xbrl(fiscal_year=2023, identificador="b"),
        ]
        for derivada in (20.0, 25.0):  # diferencia y variación porcentual
            with self.subTest(derivada=derivada):
                self.assertIsNone(
                    self._ejecutar(
                        _respuesta(cifra=derivada, unidad="USD"), mensajes
                    )
                )

    def test_corrige_una_sola_vez(self) -> None:
        primera = self._ejecutar(_respuesta(cifra=250.0, unidad="USD"))
        assert primera is not None
        mensajes: list[object] = [primera["messages"][0]]
        self.assertIsNone(
            self._ejecutar(_respuesta(cifra=250.0, unidad="USD"), mensajes)
        )

    def test_rechaza_un_corpus_que_no_lo_es(self) -> None:
        with self.assertRaises(TypeError):
            ComprobadorCifrasXBRL("corpus")  # type: ignore[arg-type]


class TestReparadorCitas(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.corpus, self.fragmentos = crear_corpus_temporal(
            Path(self.temporal.name)
        )
        self.reparador = ReparadorCitas(self.corpus)
        self.runtime = Runtime()
        self.fragmento = self.fragmentos[0]

    def tearDown(self) -> None:
        self.temporal.cleanup()

    def _ejecutar(
        self,
        respuesta: RespuestaFinanciera,
        mensajes: list[object] | None = None,
    ) -> dict | None:
        return self.reparador.after_model(
            {
                "messages": mensajes
                if mensajes is not None
                else _busqueda(self.fragmento.chunk_id, self.fragmento.texto),
                "structured_response": respuesta,
            },
            self.runtime,
        )

    def test_no_toca_una_cita_ya_valida(self) -> None:
        respuesta = _respuesta(
            cita="Supply chain disruptions",
            chunk_id=self.fragmento.chunk_id,
        )
        self.assertIsNone(self._ejecutar(respuesta))

    def test_quita_las_comillas_que_anade_el_modelo(self) -> None:
        respuesta = _respuesta(
            cita=f"«{TEXTO_2024}»",
            chunk_id=self.fragmento.chunk_id,
        )
        resultado = self._ejecutar(respuesta)
        assert resultado is not None
        self.assertEqual(resultado["structured_response"].cita, TEXTO_2024)

    def test_repara_tipografia_espacios_y_elipsis(self) -> None:
        cita = "supply chain    disruptions ... harm operations"
        resultado = self._ejecutar(
            _respuesta(cita=cita, chunk_id=self.fragmento.chunk_id)
        )
        assert resultado is not None
        reparada = resultado["structured_response"].cita
        self.assertIn(reparada, self.fragmento.texto)
        self.assertTrue(reparada.startswith("Supply chain"))

    def test_corrige_el_chunk_id_cuando_la_cita_esta_en_otro_fragmento(self) -> None:
        otro = self.fragmentos[1]
        respuesta = _respuesta(
            cita=otro.texto,
            chunk_id=self.fragmento.chunk_id,
        )
        resultado = self._ejecutar(
            respuesta, _busqueda(otro.chunk_id, otro.texto)
        )
        assert resultado is not None
        self.assertEqual(
            resultado["structured_response"].chunk_id, otro.chunk_id
        )

    def test_no_inventa_una_cita_que_no_esta_en_el_fragmento(self) -> None:
        respuesta = _respuesta(
            cita="Las interrupciones de la cadena de suministro dañan el negocio.",
            chunk_id=self.fragmento.chunk_id,
        )
        self.assertIsNone(self._ejecutar(respuesta))

    def test_no_actua_sin_fragmentos_recuperados_ni_citados(self) -> None:
        respuesta = _respuesta(cita=TEXTO_2024, chunk_id="INEXISTENTE-2024-1A-0000")
        self.assertIsNone(self._ejecutar(respuesta, []))


class TestUtilidadesDeTraza(unittest.TestCase):
    def test_chunks_recuperados_conserva_el_orden_sin_repetir(self) -> None:
        mensajes = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_filings", "args": {}, "id": "call-busqueda"}
                ],
            ),
            ToolMessage(
                content=(
                    "[ACME-2024-1A-0000] ACME FY2024 Item 1A\ntexto\n\n---\n\n"
                    "[ACME-2023-7-0000] ACME FY2023 Item 7\ntexto"
                ),
                tool_call_id="call-busqueda",
                name="search_filings",
            ),
        ]
        self.assertEqual(
            chunks_recuperados(mensajes),
            ("ACME-2024-1A-0000", "ACME-2023-7-0000"),
        )
        self.assertEqual(chunks_recuperados(None), ())

    def test_buscar_literal_exige_un_minimo_y_no_cose_trozos_lejanos(self) -> None:
        texto = "Alpha beta gamma. " + "relleno " * 80 + "delta epsilon zeta."
        self.assertIsNone(buscar_literal("Alpha beta", texto))
        self.assertIsNone(
            buscar_literal("Alpha beta gamma ... delta epsilon zeta", texto)
        )


class TestRegistroDeIntervenciones(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temporal.name)
        self.corpus, self.fragmentos = crear_corpus_temporal(self.raiz / "c")
        self.runtime = Runtime()

    def tearDown(self) -> None:
        self.temporal.cleanup()

    def test_registra_la_correccion_con_su_pregunta(self) -> None:
        comprobador = ComprobadorCifrasXBRL(self.corpus)
        comprobador.after_model(
            {
                "messages": [HumanMessage("¿Beneficio de ACME en 2024?")],
                "structured_response": _respuesta(cifra=250.0, unidad="USD"),
            },
            self.runtime,
        )
        (intervencion,) = comprobador.intervenciones
        self.assertEqual(intervencion["tipo"], "correccion")
        self.assertEqual(intervencion["pregunta"], "¿Beneficio de ACME en 2024?")
        self.assertIn(MARCA_CIFRAS, intervencion["mensaje"])

    def test_registra_la_cita_reparada(self) -> None:
        reparador = ReparadorCitas(self.corpus)
        fragmento = self.fragmentos[0]
        reparador.after_model(
            {
                "messages": _busqueda(fragmento.chunk_id, fragmento.texto),
                "structured_response": _respuesta(
                    cita=f'"{TEXTO_2024}"', chunk_id=fragmento.chunk_id
                ),
            },
            self.runtime,
        )
        (intervencion,) = reparador.intervenciones
        self.assertEqual(intervencion["tipo"], "cita_reparada")
        self.assertEqual(intervencion["despues"], TEXTO_2024)

    def test_vuelca_las_intervenciones_de_todos_los_middlewares(self) -> None:
        comprobador = ComprobadorCifrasXBRL(self.corpus)
        comprobador.intervenciones.append({"tipo": "correccion"})
        reparador = ReparadorCitas(self.corpus)
        reparador.intervenciones.append({"tipo": "cita_reparada"})
        constructor = type("C", (), {"middlewares": (comprobador, reparador)})()
        ruta = self.raiz / "salida" / "intervenciones.json"

        self.assertEqual(volcar_intervenciones(constructor, ruta), 2)
        self.assertEqual(
            [i["tipo"] for i in json.loads(ruta.read_text(encoding="utf-8"))],
            ["correccion", "cita_reparada"],
        )

    def test_sin_middlewares_no_escribe_nada(self) -> None:
        constructor = type("C", (), {"middlewares": ()})()
        ruta = self.raiz / "nada.json"
        self.assertEqual(volcar_intervenciones(constructor, ruta), 0)
        self.assertFalse(ruta.exists())


if __name__ == "__main__":
    unittest.main()
