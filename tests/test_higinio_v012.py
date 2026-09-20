from __future__ import annotations

import unittest

from experimentos.higinio.agente_v011 import crear_constructor as crear_v011
from experimentos.higinio.agente_v012 import (
    INSTRUCCIONES_COMPARATIVAS,
    NOMBRE_VARIANTE,
    crear_constructor as crear_v012,
)


class TestHiginioV012(unittest.TestCase):
    def test_parte_de_v011_y_solo_amplia_el_prompt(self) -> None:
        v011 = crear_v011()
        v012 = crear_v012()

        self.assertEqual(v012.nombre, NOMBRE_VARIANTE)
        self.assertEqual(
            v012.configuracion.system_prompt,
            (
                f"{v011.configuracion.system_prompt}\n\n"
                f"{INSTRUCCIONES_COMPARATIVAS}"
            ),
        )
        self.assertEqual(
            v012.configuracion.model_dump(exclude={"system_prompt"}),
            v011.configuracion.model_dump(exclude={"system_prompt"}),
        )
        self.assertEqual(v012.corpus, v011.corpus)
        self.assertIs(v012.esquema_respuesta, v011.esquema_respuesta)
        self.assertEqual(
            tuple(type(middleware) for middleware in v012.middlewares),
            tuple(type(middleware) for middleware in v011.middlewares),
        )
        self.assertIs(v012.modelo, v011.modelo)
        self.assertEqual(
            dict(v012.control_peticiones.parametros),
            dict(v011.control_peticiones.parametros),
        )
        self.assertIs(v012.telemetria_auxiliar, v011.telemetria_auxiliar)

    def test_prompt_acota_busquedas_y_fija_la_cifra_final(self) -> None:
        prompt = INSTRUCCIONES_COMPARATIVAS

        self.assertIn("get_xbrl_fact", prompt)
        self.assertIn("search_filings", prompt)
        self.assertIn("cita y chunk_id no estén vacíos", prompt)
        self.assertIn("ejercicio final", prompt)
        self.assertIn("máximo una vez", prompt)
        self.assertIn("no la repitas", prompt)


if __name__ == "__main__":
    unittest.main()
