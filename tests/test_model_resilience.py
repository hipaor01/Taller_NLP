from __future__ import annotations

import unittest

from taller_nlp import ControlPeticionesModelo, RateLimitAgotadoError


class ErrorHTTP(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestControlPeticionesModelo(unittest.TestCase):
    def test_reintenta_429_con_backoff_y_jitter_acotado(self) -> None:
        esperas: list[float] = []
        intentos = 0
        control = ControlPeticionesModelo(
            solicitudes_por_minuto=None,
            max_reintentos=3,
            espera_inicial_s=5,
            factor_espera=2,
            espera_maxima_s=20,
            jitter_s=0.5,
            dormir=esperas.append,
            generar_jitter=lambda _inicio, _fin: 0.25,
        )

        def operacion() -> str:
            nonlocal intentos
            intentos += 1
            if intentos < 4:
                raise ErrorHTTP(429)
            return "ok"

        self.assertEqual(control.ejecutar(operacion), "ok")
        self.assertEqual(intentos, 4)
        self.assertEqual(esperas, [5.25, 10.25, 20])

    def test_no_reintenta_errores_que_no_sean_429(self) -> None:
        control = ControlPeticionesModelo(
            solicitudes_por_minuto=None,
            dormir=lambda _: self.fail("No debía esperar"),
        )
        with self.assertRaisesRegex(ErrorHTTP, "HTTP 401"):
            control.ejecutar(lambda: (_ for _ in ()).throw(ErrorHTTP(401)))

    def test_informa_al_agotar_reintentos(self) -> None:
        control = ControlPeticionesModelo(
            solicitudes_por_minuto=None,
            max_reintentos=1,
            espera_inicial_s=0,
            espera_maxima_s=0,
            jitter_s=0,
            dormir=lambda _: None,
        )
        with self.assertRaises(RateLimitAgotadoError) as contexto:
            control.ejecutar(lambda: (_ for _ in ()).throw(ErrorHTTP(429)))
        self.assertIsInstance(contexto.exception.__cause__, ErrorHTTP)

    def test_crea_limitador_sin_rafagas(self) -> None:
        control = ControlPeticionesModelo(solicitudes_por_minuto=18)
        self.assertIsNotNone(control.limitador)
        self.assertAlmostEqual(
            control.limitador.requests_per_second, 0.3  # type: ignore[union-attr]
        )
        self.assertEqual(
            control.limitador.max_bucket_size, 1  # type: ignore[union-attr]
        )


if __name__ == "__main__":
    unittest.main()
