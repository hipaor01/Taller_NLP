from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage

from experimentos.higinio.agente_v001 import crear_constructor as crear_v001
from experimentos.higinio.agente_v003 import (
    NOMBRE_VARIANTE,
    crear_constructor as crear_v003,
)
from taller_nlp import (
    ControlPeticionesModelo,
    ReescritorConsultaLLM,
    RegistroTelemetriaAuxiliar,
    RetrieverConReescritura,
)
from tests.support import (
    RetrieverControlado,
    como_recuperado,
    crear_corpus_temporal,
)


class TestReescrituraConsulta(unittest.TestCase):
    def test_reescribe_antes_de_buscar_y_registra_telemetria(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            corpus, fragmentos = crear_corpus_temporal(Path(temporal))
            denso = RetrieverControlado(corpus, (como_recuperado(fragmentos[0]),))
            registro = RegistroTelemetriaAuxiliar()
            reescritor = ReescritorConsultaLLM(
                nombre_modelo="modelo-prueba",
                instruccion="Traduce al inglés.",
                modelo=FakeMessagesListChatModel(
                    responses=[
                        AIMessage(
                            content="underestimated customer demand",
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "total_tokens": 12,
                            },
                        )
                    ]
                ),
                control_peticiones=ControlPeticionesModelo(
                    solicitudes_por_minuto=None
                ),
                telemetria=registro,
                precio_entrada_usd_millon_tokens=0.75,
                precio_salida_usd_millon_tokens=3.75,
            )
            retriever = RetrieverConReescritura(
                corpus,
                retriever_base=denso,
                reescritor=reescritor,
            )

            with registro.capturar() as captura:
                resultados = retriever.buscar("demanda subestimada", k=1)

            self.assertEqual(resultados[0].chunk_id, fragmentos[0].chunk_id)
            assert denso.ultima_entrada is not None
            self.assertEqual(
                denso.ultima_entrada["query"],
                "underestimated customer demand",
            )
            self.assertEqual(len(captura.llamadas), 1)
            self.assertEqual(
                captura.llamadas[0].componente,
                "reescritor-consulta-llm",
            )
            self.assertEqual(captura.llamadas[0].tokens_entrada, 10)
            self.assertEqual(captura.llamadas[0].tokens_salida, 2)
            self.assertAlmostEqual(
                captura.llamadas[0].coste_usd or 0,
                0.000015,
            )

    def test_sin_modelo_usa_el_respaldo_o_la_consulta_original(self) -> None:
        reescritor = ReescritorConsultaLLM(
            nombre_modelo="modelo-prueba",
            instruccion="Traduce.",
            modelo=None,
            control_peticiones=ControlPeticionesModelo(
                solicitudes_por_minuto=None
            ),
            telemetria=RegistroTelemetriaAuxiliar(),
            reescrituras_respaldo={"q-1": "recorded rewrite"},
        )

        self.assertEqual(
            reescritor.reescribir("pregunta", id_golden="q-1"),
            "recorded rewrite",
        )
        self.assertEqual(reescritor.reescribir("pregunta"), "pregunta")


class TestHiginioV003(unittest.TestCase):
    def test_parte_de_v001_y_no_del_hibrido_v002(self) -> None:
        v001 = crear_v001()
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("OPENROUTER_API_KEY", None)
            v003 = crear_v003()
        retriever_v001 = v001.fabrica_herramientas.retriever
        retriever_v003 = v003.fabrica_herramientas.retriever

        self.assertEqual(v003.nombre, NOMBRE_VARIANTE)
        self.assertEqual(v003.configuracion, v001.configuracion)
        self.assertEqual(v003.corpus, v001.corpus)
        self.assertEqual(v003.evaluador.k_retrieval, v001.evaluador.k_retrieval)
        self.assertIsInstance(retriever_v003, RetrieverConReescritura)
        assert isinstance(retriever_v003, RetrieverConReescritura)
        self.assertIsNotNone(retriever_v001)
        assert retriever_v001 is not None
        self.assertEqual(retriever_v003.retriever_base.nombre, retriever_v001.nombre)
        self.assertTrue(retriever_v003.retriever_base.aplica_filtros_metadatos)
        self.assertFalse(retriever_v003.reescritor.modelo_disponible)
        self.assertIsNotNone(v003.telemetria_auxiliar)

    def test_con_clave_crea_el_reescritor_con_el_limitador_compartido(self) -> None:
        modelo = FakeListChatModel(responses=["rewritten query"])
        with patch.dict(
            "os.environ",
            {"OPENROUTER_API_KEY": "clave-de-prueba"},
        ):
            with patch(
                "experimentos.higinio.agente_v003.crear_modelo_chat",
                return_value=modelo,
            ) as crear_modelo:
                constructor = crear_v003()

        retriever = constructor.fabrica_herramientas.retriever
        assert isinstance(retriever, RetrieverConReescritura)
        self.assertTrue(retriever.reescritor.modelo_disponible)
        self.assertIs(
            crear_modelo.call_args.kwargs["limitador"],
            constructor.control_peticiones.limitador,
        )


if __name__ == "__main__":
    unittest.main()
