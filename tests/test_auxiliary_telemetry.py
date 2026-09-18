from __future__ import annotations

import unittest

from taller_nlp import LlamadaModeloAuxiliar, RegistroTelemetriaAuxiliar


class TestRegistroTelemetriaAuxiliar(unittest.TestCase):
    def test_captura_solo_las_llamadas_de_la_ejecucion_activa(self) -> None:
        registro = RegistroTelemetriaAuxiliar()
        llamada = LlamadaModeloAuxiliar(
            componente="reescritor_consulta",
            modelo="modelo",
            latencia_ms=12,
            tokens_entrada=8,
            tokens_salida=3,
            coste_usd=0.001,
        )

        self.assertFalse(registro.registrar(llamada))
        with registro.capturar() as captura:
            self.assertTrue(registro.registrar(llamada))
            self.assertEqual(captura.llamadas, (llamada,))
        self.assertFalse(registro.registrar(llamada))

    def test_capturas_consecutivas_no_mezclan_llamadas(self) -> None:
        registro = RegistroTelemetriaAuxiliar()
        primera = LlamadaModeloAuxiliar(
            componente="primera",
            modelo="modelo",
            latencia_ms=1,
        )
        segunda = LlamadaModeloAuxiliar(
            componente="segunda",
            modelo="modelo",
            latencia_ms=2,
        )

        with registro.capturar() as captura_1:
            registro.registrar(primera)
        with registro.capturar() as captura_2:
            registro.registrar(segunda)

        self.assertEqual(captura_1.llamadas, (primera,))
        self.assertEqual(captura_2.llamadas, (segunda,))

    def test_rechaza_objetos_que_no_sean_llamadas_normalizadas(self) -> None:
        registro = RegistroTelemetriaAuxiliar()
        with self.assertRaisesRegex(TypeError, "LlamadaModeloAuxiliar"):
            registro.registrar(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
