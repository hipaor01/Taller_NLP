from __future__ import annotations

import unittest
from unittest.mock import patch

from experimentos import baseline
from experimentos.jchulvi import agente_v2, agente_v3


class TestAgenteJchulviV3(unittest.TestCase):
    def test_sustituye_la_regla_incompatible_sin_mutar_v2(self) -> None:
        self.assertIn(agente_v3.REGLA_AUSENCIA_XBRL, agente_v3.INSTRUCCIONES)
        self.assertNotIn(agente_v3._REGLA_TEXTO_V2, agente_v3.INSTRUCCIONES)
        self.assertIn(agente_v3._REGLA_TEXTO_V2, agente_v2.INSTRUCCIONES)
        self.assertIn("fuente=\"ninguna\"", agente_v3.INSTRUCCIONES)
        self.assertIn("NO calcules ni derives", agente_v3.INSTRUCCIONES)
        self.assertIn("NO consultes otros conceptos", agente_v3.INSTRUCCIONES)

    def test_conserva_la_composicion_de_v2_y_cambia_solo_el_prompt(self) -> None:
        base = baseline.crear_constructor_baseline().con_ruta_progreso(
            "progreso-v3.json"
        )

        with patch.object(
            agente_v3.agente_v2,
            "crear_constructor",
            return_value=base,
        ) as crear_v2:
            constructor = agente_v3.crear_constructor(
                ruta_progreso="progreso-v3.json"
            )

        crear_v2.assert_called_once_with(ruta_progreso="progreso-v3.json")
        self.assertEqual(constructor.nombre, agente_v3.NOMBRE)
        self.assertEqual(
            constructor.configuracion.system_prompt,
            agente_v3.INSTRUCCIONES,
        )
        self.assertEqual(
            constructor.configuracion.model_dump(
                exclude={"system_prompt"}
            ),
            base.configuracion.model_dump(exclude={"system_prompt"}),
        )
        self.assertIs(constructor.corpus, base.corpus)
        self.assertIs(
            constructor.fabrica_herramientas,
            base.fabrica_herramientas,
        )
        self.assertEqual(constructor.middlewares, base.middlewares)
        self.assertIs(constructor.modelo, base.modelo)
        self.assertIs(
            constructor.control_peticiones,
            base.control_peticiones,
        )
        self.assertEqual(
            constructor.evaluador.ruta_progreso,
            base.evaluador.ruta_progreso,
        )


if __name__ == "__main__":
    unittest.main()
