from __future__ import annotations

import unittest

from experimentos.higinio.agente_v015 import crear_constructor as crear_v015
from experimentos.higinio.agente_v017 import (
    INSTRUCCIONES_AUSENCIA_XBRL,
    NOMBRE_VARIANTE,
    crear_constructor as crear_v017,
)


class TestHiginioV017(unittest.TestCase):
    def test_parte_de_v015_y_solo_amplia_el_prompt(self) -> None:
        v015 = crear_v015()
        v017 = crear_v017()

        self.assertEqual(v017.nombre, NOMBRE_VARIANTE)
        self.assertEqual(
            v017.configuracion.model_dump(exclude={"system_prompt"}),
            v015.configuracion.model_dump(exclude={"system_prompt"}),
        )
        self.assertEqual(
            v017.configuracion.system_prompt,
            (
                f"{v015.configuracion.system_prompt}\n\n"
                f"{INSTRUCCIONES_AUSENCIA_XBRL}"
            ),
        )
        self.assertEqual(v017.corpus, v015.corpus)
        retriever_v015 = v015.fabrica_herramientas.retriever
        retriever_v017 = v017.fabrica_herramientas.retriever
        self.assertIsNotNone(retriever_v015)
        self.assertIsNotNone(retriever_v017)
        assert retriever_v015 is not None
        assert retriever_v017 is not None
        self.assertIs(type(retriever_v017), type(retriever_v015))
        self.assertEqual(retriever_v017.nombre, retriever_v015.nombre)
        self.assertEqual(
            dict(retriever_v017.parametros),
            dict(retriever_v015.parametros),
        )
        self.assertIs(v017.esquema_respuesta, v015.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v017.middlewares),
            tuple(type(middleware) for middleware in v015.middlewares),
        )
        self.assertIs(v017.modelo, v015.modelo)
        self.assertEqual(
            dict(v017.control_peticiones.parametros),
            dict(v015.control_peticiones.parametros),
        )
        self.assertIs(v017.telemetria_auxiliar, v015.telemetria_auxiliar)
        self.assertEqual(
            v017.evaluador.k_retrieval,
            v015.evaluador.k_retrieval,
        )

    def test_prompt_prohibe_sustituir_derivar_y_buscar_en_texto(self) -> None:
        prompt = crear_v017().configuracion.system_prompt

        for instruccion in (
            "considera esa respuesta definitiva",
            "NO uses search_filings ni read_section",
            "NO consultes otros conceptos",
            "NO calcules ni derives la magnitud",
            "NO incluyas en la respuesta cifras de conceptos alternativos",
            "cifra=null, unidad=null",
            'fuente="ninguna", cita=null y chunk_id=null',
        ):
            with self.subTest(instruccion=instruccion):
                self.assertIn(instruccion, prompt)


if __name__ == "__main__":
    unittest.main()
