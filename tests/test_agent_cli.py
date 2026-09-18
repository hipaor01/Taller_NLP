from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agente.__main__ import main
from agente.interfaz import VARIABLE_VARIANTE


class TestCliAgente(unittest.TestCase):
    def test_evalua_guarda_y_selecciona_variante(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            entrada = raiz / "holdout.jsonl"
            entrada.write_text("{}\n", encoding="utf-8")
            salida = raiz / "resultados" / "final.csv"
            tabla = pd.DataFrame([{"id": "q1", "trayectoria": True}])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(VARIABLE_VARIANTE, None)
                with patch(
                    "agente.__main__.evaluar",
                    return_value=tabla,
                ) as evaluar_mock:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        codigo = main(
                            [
                                "--evaluar",
                                str(entrada),
                                "--salida",
                                str(salida),
                                "--variante",
                                "experimentos.equipo.agente_v002",
                            ]
                        )

                self.assertEqual(
                    os.environ[VARIABLE_VARIANTE],
                    "experimentos.equipo.agente_v002",
                )

            self.assertEqual(codigo, 0)
            evaluar_mock.assert_called_once_with(entrada, salida=salida)
            self.assertIn("q1", stdout.getvalue())
            self.assertIn(str(salida.resolve()), stderr.getvalue())

    def test_exige_ruta_de_evaluacion(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as contexto:
                main([])
        self.assertEqual(contexto.exception.code, 2)

    def test_resume_csv_con_etiqueta_deducida(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "baseline.csv"
            pd.DataFrame(
                {
                    "cita": [True, False],
                    "coste_usd": [0.01, 0.03],
                    "latencia_s": [1.0, 3.0],
                    "llamadas": [2, 4],
                }
            ).to_csv(ruta, index=False)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                codigo = main(["--resumir", str(ruta)])

            self.assertEqual(codigo, 0)
            impreso = stdout.getvalue()
            self.assertIn("baseline", impreso)
            self.assertIn("coste medio (¢)", impreso)

    def test_resume_csv_con_etiqueta_y_salida_explicitas(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            entrada = raiz / "evaluacion.csv"
            salida = raiz / "informes" / "resumen.csv"
            pd.DataFrame({"trayectoria": [True, False]}).to_csv(
                entrada,
                index=False,
            )

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                codigo = main(
                    [
                        "--resumir",
                        str(entrada),
                        "--etiqueta",
                        "baseline",
                        "--salida",
                        str(salida),
                    ]
                )

            self.assertEqual(codigo, 0)
            resumen = pd.read_csv(salida)
            self.assertEqual(resumen.loc[0, "versión"], "baseline")
            self.assertEqual(resumen.loc[0, "trayectoria"], 0.5)

    def test_rechaza_opciones_de_otra_operacion(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "baseline.csv"
            pd.DataFrame({"cita": [True]}).to_csv(ruta, index=False)
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as contexto:
                    main(
                        [
                            "--resumir",
                            str(ruta),
                            "--variante",
                            "experimentos.baseline",
                        ]
                    )
        self.assertEqual(contexto.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
