from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from taller_nlp import CorteFragmento, FragmentoCorpus, TroceadorCorpus

from tests.support import IndexadorTexto, crear_fragmento


class TroceadorCortes(TroceadorCorpus):
    def __init__(self, cortes: tuple[CorteFragmento, ...]) -> None:
        super().__init__("cortes", parametros={"nested": {"value": 1}})
        self._cortes = cortes

    def _calcular_cortes(
        self, *, texto: str, ticker: str, fiscal_year: int, item: str
    ) -> Iterable[CorteFragmento]:
        del texto, ticker, fiscal_year, item
        return self._cortes


class TestTroceadorCorpus(unittest.TestCase):
    def test_construye_texto_literal_ids_estables_y_admite_solape(self) -> None:
        troceador = TroceadorCortes(
            (
                CorteFragmento(inicio_car=0, fin_car=6, n_tokens=1),
                CorteFragmento(inicio_car=4, fin_car=10, n_tokens=2),
            )
        )
        fragmentos = troceador.trocear(
            ticker=" acme ", fiscal_year=2024, item=" 1a ", texto="abcdefghij"
        )
        self.assertEqual([f.texto for f in fragmentos], ["abcdef", "efghij"])
        self.assertEqual(
            [f.chunk_id for f in fragmentos],
            ["ACME-2024-1A-0000", "ACME-2024-1A-0001"],
        )

    def test_rechaza_huecos_con_texto_no_blanco(self) -> None:
        troceador = TroceadorCortes(
            (
                CorteFragmento(inicio_car=0, fin_car=2, n_tokens=1),
                CorteFragmento(inicio_car=4, fin_car=6, n_tokens=1),
            )
        )
        with self.assertRaisesRegex(ValueError, "texto no vacío"):
            troceador.trocear(
                ticker="ACME", fiscal_year=2024, item="1A", texto="abcdef"
            )

    def test_rechaza_cortes_duplicados_desordenados_o_fuera(self) -> None:
        casos = (
            (
                CorteFragmento(inicio_car=0, fin_car=3, n_tokens=1),
                CorteFragmento(inicio_car=0, fin_car=3, n_tokens=1),
            ),
            (
                CorteFragmento(inicio_car=2, fin_car=6, n_tokens=1),
                CorteFragmento(inicio_car=0, fin_car=3, n_tokens=1),
            ),
            (CorteFragmento(inicio_car=0, fin_car=7, n_tokens=1),),
        )
        for cortes in casos:
            with self.subTest(cortes=cortes), self.assertRaises(ValueError):
                TroceadorCortes(cortes).trocear(
                    ticker="ACME", fiscal_year=2024, item="1A", texto="abcdef"
                )

    def test_parametros_se_exponen_como_copia_de_solo_lectura(self) -> None:
        troceador = TroceadorCortes(
            (CorteFragmento(inicio_car=0, fin_car=1, n_tokens=1),)
        )
        parametros = troceador.parametros
        parametros["nested"]["value"] = 9
        self.assertEqual(troceador.parametros["nested"]["value"], 1)
        with self.assertRaises(TypeError):
            parametros["nuevo"] = 2

    def test_fragmento_rechaza_id_incoherente_con_metadatos(self) -> None:
        with self.assertRaisesRegex(ValidationError, "no coincide"):
            FragmentoCorpus(
                chunk_id="ACME-2024-1A-0001",
                ticker="ACME",
                fiscal_year=2024,
                item="1A",
                posicion=0,
                texto="abc",
                n_tokens=1,
                contiene_tabla=False,
                inicio_car=0,
                fin_car=3,
            )


class TestIndexadorCorpus(unittest.TestCase):
    def test_liga_artefactos_a_parametros_hash_y_orden_de_fragmentos(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            salida = Path(temporal)
            fragmentos = (
                crear_fragmento("abc", posicion=0),
                crear_fragmento("def", posicion=1, inicio_car=3),
            )
            indexador = IndexadorTexto()
            artefactos = indexador.indexar(fragmentos, salida)
            self.assertEqual(indexador.fragmentos_recibidos, fragmentos)
            self.assertEqual(artefactos.numero_fragmentos, 2)
            self.assertEqual(artefactos.parametros_indexacion, {"version": 1})
            self.assertTrue(artefactos.indice.ruta.is_file())
            self.assertTrue(artefactos.metadatos.ruta.is_file())

    def test_rechaza_entrada_vacia_y_posiciones_no_contiguas(self) -> None:
        with tempfile.TemporaryDirectory() as temporal:
            indexador = IndexadorTexto()
            with self.assertRaisesRegex(ValueError, "secuencia vacía"):
                indexador.indexar((), Path(temporal))
            with self.assertRaisesRegex(ValueError, "no son contiguas"):
                indexador.indexar(
                    (crear_fragmento("abc", posicion=1),), Path(temporal)
                )


if __name__ == "__main__":
    unittest.main()
