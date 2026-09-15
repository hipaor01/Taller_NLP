"""Persistencia atómica de resultados parciales de una evaluación."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import ResultadoPregunta


class ContextoProgreso(BaseModel):
    """Identidad que impide mezclar resultados de evaluaciones distintas."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    nombre_agente: str = Field(min_length=1)
    sha256_golden: str = Field(pattern=r"^[0-9a-f]{64}$")
    firma_evaluacion: str = Field(pattern=r"^[0-9a-f]{64}$")
    ids_casos: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validar_ids_unicos(self) -> "ContextoProgreso":
        if len(self.ids_casos) != len(set(self.ids_casos)):
            raise ValueError("Los identificadores de casos deben ser únicos.")
        return self


class AlmacenProgresoEvaluacion:
    """Guarda cada caso mediante reemplazo atómico y permite reanudarlo."""

    def __init__(self, ruta: str | Path) -> None:
        self._ruta = Path(ruta).resolve()
        self._contexto: ContextoProgreso | None = None
        self._resultados: dict[str, ResultadoPregunta] = {}

    @property
    def ruta(self) -> Path:
        return self._ruta

    def preparar(
        self, contexto: ContextoProgreso
    ) -> dict[str, ResultadoPregunta]:
        """Carga un progreso compatible o comienza uno todavía vacío."""
        self._contexto = contexto
        self._resultados = {}
        if not self._ruta.exists():
            return {}
        if not self._ruta.is_file():
            raise ValueError(f"La ruta de progreso no es un fichero: {self._ruta}")
        try:
            datos = json.loads(self._ruta.read_text(encoding="utf-8"))
            contexto_guardado = ContextoProgreso.model_validate(datos["contexto"])
            resultados = tuple(
                ResultadoPregunta.model_validate(resultado)
                for resultado in datos["resultados"]
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"El progreso de evaluación no es válido: {self._ruta}"
            ) from exc
        if contexto_guardado != contexto:
            raise ValueError(
                "El fichero de progreso pertenece a otra configuración, "
                "agente o versión del golden set. Usa otra ruta o reinícialo."
            )

        permitidos = set(contexto.ids_casos)
        for resultado in resultados:
            if resultado.id_pregunta not in permitidos:
                raise ValueError(
                    "El progreso contiene una pregunta ajena al golden set: "
                    f"{resultado.id_pregunta}"
                )
            if resultado.id_pregunta in self._resultados:
                raise ValueError(
                    "El progreso contiene un id_pregunta duplicado: "
                    f"{resultado.id_pregunta}"
                )
            self._resultados[resultado.id_pregunta] = resultado
        return dict(self._resultados)

    def registrar(self, resultado: ResultadoPregunta) -> None:
        """Añade un resultado terminado y sincroniza el fichero atómicamente."""
        if self._contexto is None:
            raise RuntimeError("Debe llamarse preparar() antes de registrar().")
        if resultado.id_pregunta not in self._contexto.ids_casos:
            raise ValueError("El resultado no pertenece a esta evaluación.")
        if resultado.id_pregunta in self._resultados:
            raise ValueError(
                f"La pregunta {resultado.id_pregunta} ya estaba registrada."
            )
        self._resultados[resultado.id_pregunta] = resultado
        self._guardar()

    def _guardar(self) -> None:
        assert self._contexto is not None
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = {
            "contexto": self._contexto.model_dump(mode="json"),
            "resultados": [
                resultado.model_dump(
                    mode="json", exclude_computed_fields=True
                )
                for resultado in self._resultados.values()
            ],
        }
        with TemporaryDirectory(
            prefix=".progreso-evaluacion-", dir=self._ruta.parent
        ) as temporal:
            provisional = Path(temporal) / self._ruta.name
            provisional.write_text(
                json.dumps(datos, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            provisional.replace(self._ruta)
