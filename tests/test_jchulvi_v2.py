from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from experimentos import baseline
from experimentos.jchulvi import agente_v2, qdrant_local


class _QdrantControlado:
    url = "http://qdrant.test"
    coleccion = "coleccion-prueba"

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.eventos: list[str] = []

    @contextmanager
    def sesion(self):
        self.eventos.append("abrir")
        try:
            yield self
        finally:
            self.eventos.append("cerrar")


class TestAgenteJchulviV2(unittest.TestCase):
    def test_conserva_composicion_y_declara_qdrant_como_recurso(self) -> None:
        infraestructura = baseline.crear_constructor_baseline()
        qdrant = _QdrantControlado()

        with patch.object(
            agente_v2.qdrant_local,
            "QdrantLocal",
            return_value=qdrant,
        ), patch.object(
            agente_v2.agente,
            "crear_constructor",
            return_value=infraestructura,
        ) as crear_infraestructura:
            constructor = agente_v2.crear_constructor(
                ruta_progreso="progreso.json"
            )

        crear_infraestructura.assert_called_once_with(
            modo="denso",
            guardrail=False,
            qdrant_url=qdrant.url,
            coleccion=qdrant.coleccion,
            ruta_progreso="progreso.json",
        )
        self.assertEqual(constructor.nombre, agente_v2.NOMBRE)
        self.assertEqual(constructor.middlewares, ())
        self.assertEqual(constructor.configuracion.max_iteraciones, 6)
        self.assertEqual(
            constructor.configuracion.system_prompt.strip(),
            agente_v2.INSTRUCCIONES.strip(),
        )
        self.assertEqual(len(constructor.recursos_ejecucion), 1)

        with constructor.sesion_ejecucion():
            self.assertEqual(qdrant.eventos, ["abrir"])
        self.assertEqual(qdrant.eventos, ["abrir", "cerrar"])

    def test_sesiones_qdrant_anidadas_comparten_el_contenedor(self) -> None:
        primero = object.__new__(qdrant_local.QdrantLocal)
        segundo = object.__new__(qdrant_local.QdrantLocal)
        primero.iniciar = Mock()
        primero.cargar = Mock()
        primero.detener = Mock()
        segundo.iniciar = Mock()
        segundo.cargar = Mock()
        segundo.detener = Mock()

        with primero.sesion():
            with segundo.sesion():
                primero.detener.assert_not_called()
                segundo.detener.assert_not_called()

        primero.iniciar.assert_called_once_with()
        segundo.iniciar.assert_not_called()
        primero.cargar.assert_called_once_with()
        segundo.cargar.assert_called_once_with()
        primero.detener.assert_called_once_with()
        segundo.detener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
