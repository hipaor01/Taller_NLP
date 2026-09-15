"""Colección validada de herramientas intercambiables del agente."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Mapping

from langchain_core.tools import BaseTool


_CONTRATOS = {
    "list_available": {
        "campos": frozenset(),
        "requeridos": frozenset(),
    },
    "get_xbrl_fact": {
        "campos": frozenset({"ticker", "fiscal_year", "concept"}),
        "requeridos": frozenset({"ticker", "fiscal_year", "concept"}),
    },
    "search_filings": {
        "campos": frozenset({"query", "ticker", "fiscal_year", "item", "k"}),
        "requeridos": frozenset({"query"}),
    },
    "read_section": {
        "campos": frozenset({"ticker", "fiscal_year", "item"}),
        "requeridos": frozenset({"ticker", "fiscal_year", "item"}),
    },
}


@dataclass(frozen=True, slots=True)
class ToolSuite:
    """Las cuatro tools del agente con nombres y argumentos estables."""

    list_available: BaseTool
    get_xbrl_fact: BaseTool
    search_filings: BaseTool
    read_section: BaseTool

    def __post_init__(self) -> None:
        for campo in fields(self):
            nombre_esperado = campo.name
            herramienta = getattr(self, nombre_esperado)
            self._validar_herramienta(nombre_esperado, herramienta)

    @staticmethod
    def _validar_herramienta(
        nombre_esperado: str, herramienta: BaseTool
    ) -> None:
        if not isinstance(herramienta, BaseTool):
            raise TypeError(
                f"{nombre_esperado} debe ser una herramienta de LangChain."
            )
        if herramienta.name != nombre_esperado:
            raise ValueError(
                f"La tool del campo {nombre_esperado} se llama "
                f"{herramienta.name!r}; debe conservar el nombre "
                f"{nombre_esperado!r}."
            )

        esquema = herramienta.get_input_schema().model_json_schema()
        campos = frozenset(esquema.get("properties", {}))
        requeridos = frozenset(esquema.get("required", ()))
        contrato = _CONTRATOS[nombre_esperado]
        if campos != contrato["campos"] or requeridos != contrato["requeridos"]:
            raise ValueError(
                f"El esquema de {nombre_esperado} no respeta el contrato. "
                f"Campos esperados={sorted(contrato['campos'])}, "
                f"requeridos={sorted(contrato['requeridos'])}; "
                f"recibidos={sorted(campos)}, "
                f"requeridos={sorted(requeridos)}."
            )

    @property
    def herramientas(self) -> tuple[BaseTool, ...]:
        """Devuelve las tools en el orden canónico usado por el agente."""
        return tuple(getattr(self, campo.name) for campo in fields(self))

    @property
    def por_nombre(self) -> Mapping[str, BaseTool]:
        """Devuelve un índice de solo lectura para ejecutar tools por nombre."""
        return MappingProxyType(
            {herramienta.name: herramienta for herramienta in self.herramientas}
        )

    def reemplazar(self, **cambios: BaseTool) -> ToolSuite:
        """Crea otra suite sustituyendo únicamente las tools indicadas."""
        nombres_validos = _CONTRATOS.keys()
        desconocidos = cambios.keys() - nombres_validos
        if desconocidos:
            raise ValueError(
                f"Herramientas desconocidas: {sorted(desconocidos)}."
            )
        return replace(self, **cambios)
