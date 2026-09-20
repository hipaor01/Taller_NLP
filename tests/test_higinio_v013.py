from __future__ import annotations

import unittest

from experimentos.higinio.agente_v012 import crear_constructor as crear_v012
from experimentos.higinio.agente_v013 import (
    INSTRUCCIONES_SELECCION_FRAGMENTO,
    NOMBRE_VARIANTE,
    crear_constructor as crear_v013,
)


class TestHiginioV013(unittest.TestCase):
    def test_parte_de_v012_y_solo_amplia_el_prompt(self) -> None:
        v012 = crear_v012()
        v013 = crear_v013()

        self.assertEqual(v013.nombre, NOMBRE_VARIANTE)
        self.assertEqual(
            v013.configuracion.system_prompt,
            (
                f"{v012.configuracion.system_prompt}\n\n"
                f"{INSTRUCCIONES_SELECCION_FRAGMENTO}"
            ),
        )
        self.assertEqual(
            v013.configuracion.model_dump(exclude={"system_prompt"}),
            v012.configuracion.model_dump(exclude={"system_prompt"}),
        )
        self.assertEqual(v013.corpus, v012.corpus)
        self.assertIs(v013.esquema_respuesta, v012.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v013.middlewares),
            tuple(type(middleware) for middleware in v012.middlewares),
        )
        self.assertEqual(
            dict(v013.control_peticiones.parametros),
            dict(v012.control_peticiones.parametros),
        )

    def test_prompt_prioriza_evidencia_de_la_comparacion(self) -> None:
        prompt = INSTRUCCIONES_SELECCION_FRAGMENTO

        self.assertIn("omite el filtro item", prompt)
        self.assertIn("incluye ambos valores", prompt)
        self.assertIn("afirmación principal", prompt)
        self.assertIn("Item 8", prompt)
        self.assertIn("Item 7", prompt)
        self.assertIn("chunk_id exacto", prompt)


if __name__ == "__main__":
    unittest.main()
