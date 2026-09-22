from __future__ import annotations

import unittest

from experimentos.higinio.agente_v017 import crear_constructor as crear_v017
from experimentos.higinio.agente_v018 import (
    INSTRUCCIONES_CIFRA_XBRL_TERMINAL,
    NOMBRE_VARIANTE,
    crear_constructor as crear_v018,
)


class TestHiginioV018(unittest.TestCase):
    def test_parte_de_v017_y_solo_amplia_el_prompt(self) -> None:
        v017 = crear_v017()
        v018 = crear_v018()

        self.assertEqual(v018.nombre, NOMBRE_VARIANTE)
        self.assertEqual(
            v018.configuracion.model_dump(exclude={"system_prompt"}),
            v017.configuracion.model_dump(exclude={"system_prompt"}),
        )
        self.assertEqual(
            v018.configuracion.system_prompt,
            (
                f"{v017.configuracion.system_prompt}\n\n"
                f"{INSTRUCCIONES_CIFRA_XBRL_TERMINAL}"
            ),
        )
        self.assertEqual(v018.corpus, v017.corpus)
        self.assertIs(v018.esquema_respuesta, v017.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v018.middlewares),
            tuple(type(middleware) for middleware in v017.middlewares),
        )
        self.assertIs(
            type(v018.fabrica_herramientas.retriever),
            type(v017.fabrica_herramientas.retriever),
        )

    def test_prompt_detiene_solo_las_preguntas_numericas_simples(self) -> None:
        prompt = " ".join(crear_v018().configuracion.system_prompt.split())

        for instruccion in (
            "Una pregunta numérica simple pide una única magnitud",
            "considera el dato definitivo y responde inmediatamente",
            "NO uses search_filings, read_section, list_available",
            'fuente="xbrl", cita=null y chunk_id=null',
            "NO se aplica a comparativas",
            "causas, explicaciones o comentarios de la dirección",
        ):
            with self.subTest(instruccion=instruccion):
                self.assertIn(instruccion, prompt)


if __name__ == "__main__":
    unittest.main()
