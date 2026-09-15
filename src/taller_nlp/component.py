"""Estado común de los componentes configurables de un experimento."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from pydantic import JsonValue


class ComponenteConfigurable:
    """Componente identificado y con parámetros JSON de solo lectura."""

    def __init__(
        self,
        nombre: str,
        *,
        parametros: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not isinstance(nombre, str):
            raise TypeError("El nombre del componente debe ser texto.")
        nombre_normalizado = nombre.strip()
        if not nombre_normalizado:
            raise ValueError("El nombre del componente no puede estar vacío.")

        parametros_recibidos = dict(parametros or {})
        if any(not isinstance(clave, str) for clave in parametros_recibidos):
            raise ValueError("Las claves de parametros deben ser texto.")
        try:
            parametros_json = json.dumps(
                parametros_recibidos,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Los parámetros deben contener valores JSON válidos."
            ) from exc

        self._nombre = nombre_normalizado
        self._parametros = json.loads(parametros_json)

    @property
    def nombre(self) -> str:
        return self._nombre

    @property
    def parametros(self) -> Mapping[str, JsonValue]:
        copia = json.loads(json.dumps(self._parametros, ensure_ascii=False))
        return MappingProxyType(copia)
