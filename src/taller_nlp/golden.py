"""Esquema y carga validada de conjuntos de evaluación JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .contracts import FamiliaPregunta, FuenteRespuesta, NombreHerramienta
from .corpus import CorpusVariant


Item10K = Literal["1A", "7", "7A", "8"]


class CasoGolden(BaseModel):
    """Una pregunta con su verdad de referencia y trayectoria esperada."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    pregunta: str = Field(min_length=1)
    familia: FamiliaPregunta
    ticker: str = Field(min_length=1)
    fiscal_year: int = Field(gt=0)
    respuesta_esperada: str = Field(min_length=1)
    cifra_esperada: float | None = Field(default=None, allow_inf_nan=False)
    unidad: str | None = None
    concept_xbrl: str | None = None
    item_esperado: Item10K | None = None
    ancla_texto: str | None = None
    ancla_inicio: int | None = Field(default=None, ge=0)
    ancla_fin: int | None = Field(default=None, ge=0)
    chunk_id_esperado: str | None = None
    herramienta_esperada: tuple[NombreHerramienta, ...] = Field(min_length=1)
    autor: str = Field(min_length=1)
    respuesta_en_corpus: bool | None = None
    fuente_esperada: FuenteRespuesta | None = None

    @field_validator("id", "pregunta", "respuesta_esperada", "autor")
    @classmethod
    def normalizar_texto_obligatorio(cls, valor: str) -> str:
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator("ticker")
    @classmethod
    def normalizar_ticker(cls, valor: str) -> str:
        normalizado = valor.strip().upper()
        if not normalizado:
            raise ValueError("ticker no puede estar vacío.")
        return normalizado

    @field_validator("unidad", "concept_xbrl", "ancla_texto", "chunk_id_esperado")
    @classmethod
    def rechazar_texto_opcional_vacio(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        if not valor.strip():
            raise ValueError("Un texto opcional debe ser None, no una cadena vacía.")
        return valor

    @field_validator("herramienta_esperada")
    @classmethod
    def validar_herramientas_sin_duplicados(
        cls, herramientas: tuple[NombreHerramienta, ...]
    ) -> tuple[NombreHerramienta, ...]:
        if len(herramientas) != len(set(herramientas)):
            raise ValueError("herramienta_esperada contiene duplicados.")
        return herramientas

    @model_validator(mode="after")
    def validar_campos_de_familia(self) -> "CasoGolden":
        es_numerica = self.familia in {"numerica", "comparativa"}
        es_textual = self.familia in {"extractiva", "comparativa"}

        campos_numericos = (self.cifra_esperada, self.unidad, self.concept_xbrl)
        if es_numerica:
            if self.concept_xbrl is None:
                raise ValueError(
                    f"La familia {self.familia} requiere concept_xbrl."
                )
            if (self.cifra_esperada is None) != (self.unidad is None):
                raise ValueError(
                    "cifra_esperada y unidad deben estar ambas informadas o "
                    "ser ambas None."
                )
        if not es_numerica and any(valor is not None for valor in campos_numericos):
            raise ValueError(
                "Una pregunta extractiva no debe contener campos numéricos."
            )

        campos_textuales = (
            self.item_esperado,
            self.ancla_texto,
            self.ancla_inicio,
            self.ancla_fin,
        )
        if es_textual and any(valor is None for valor in campos_textuales):
            raise ValueError(
                f"La familia {self.familia} requiere item, ancla y offsets."
            )
        campos_textuales_numerica = (*campos_textuales, self.chunk_id_esperado)
        if not es_textual and any(
            valor is not None for valor in campos_textuales_numerica
        ):
            raise ValueError(
                "Una pregunta numérica no debe contener campos textuales."
            )

        if self.ancla_texto is not None:
            palabras = len(self.ancla_texto.split())
            if palabras > 40:
                raise ValueError(
                    f"El ancla tiene {palabras} palabras; el máximo es 40."
                )
            assert self.ancla_inicio is not None
            assert self.ancla_fin is not None
            if self.ancla_fin <= self.ancla_inicio:
                raise ValueError("ancla_fin debe ser mayor que ancla_inicio.")
            if self.ancla_fin - self.ancla_inicio != len(self.ancla_texto):
                raise ValueError(
                    "La longitud del ancla no coincide con sus offsets."
                )
        return self

    @property
    def espera_ausencia_numerica(self) -> bool:
        """Indica que la respuesta correcta es que no hay cifra autorizada."""
        return (
            self.familia in {"numerica", "comparativa"}
            and self.cifra_esperada is None
            and self.unidad is None
        )

    def problemas_contra_corpus(
        self, secciones: pd.DataFrame, xbrl: pd.DataFrame
    ) -> list[str]:
        """Devuelve incoherencias de este caso con las fuentes autorizadas."""
        problemas = []
        disponibles = secciones[
            (secciones.ticker == self.ticker)
            & (secciones.fiscal_year.astype(int) == self.fiscal_year)
        ]
        if disponibles.empty:
            if not self.espera_ausencia_numerica:
                problemas.append(
                    f"{self.id}: {self.ticker} FY{self.fiscal_year} no está "
                    "en el corpus"
                )
            return problemas

        if self.familia in {"numerica", "comparativa"}:
            hechos = xbrl[
                (xbrl.ticker == self.ticker)
                & (xbrl.fiscal_year.astype(int) == self.fiscal_year)
                & (xbrl.concept == self.concept_xbrl)
            ]
            if hechos.empty and not self.espera_ausencia_numerica:
                problemas.append(
                    f"{self.id}: {self.ticker} no reporta "
                    f"{self.concept_xbrl!r} en FY{self.fiscal_year}"
                )
            elif not hechos.empty and self.espera_ausencia_numerica:
                problemas.append(
                    f"{self.id}: {self.ticker} sí reporta "
                    f"{self.concept_xbrl!r} en FY{self.fiscal_year}"
                )

        if self.familia in {"extractiva", "comparativa"}:
            filas = disponibles[disponibles.item == self.item_esperado]
            if len(filas) != 1:
                problemas.append(
                    f"{self.id}: se esperaría una sección Item "
                    f"{self.item_esperado}, se encontraron {len(filas)}"
                )
            else:
                texto = filas.iloc[0].texto
                assert self.ancla_inicio is not None
                assert self.ancla_fin is not None
                if texto[self.ancla_inicio : self.ancla_fin] != self.ancla_texto:
                    problemas.append(
                        f"{self.id}: los offsets no extraen el ancla esperada"
                    )
        return problemas

    @classmethod
    def validar_registros(
        cls,
        preguntas: list[dict[str, Any]],
        *,
        secciones: pd.DataFrame,
        xbrl: pd.DataFrame,
        numero_esperado: int | None = None,
        minimo_comparativas: int = 0,
    ) -> list[str]:
        """Valida una colección y devuelve todos sus problemas."""
        problemas = []
        casos = []
        vistos = set()
        for datos in preguntas:
            pid = datos.get("id", "(sin id)")
            try:
                caso = cls.model_validate(datos)
            except ValidationError as exc:
                problemas.extend(
                    f"{pid}: {error['msg']}" for error in exc.errors()
                )
                continue
            if caso.id in vistos:
                problemas.append(f"{caso.id}: id repetido")
            vistos.add(caso.id)
            casos.append(caso)
            problemas.extend(caso.problemas_contra_corpus(secciones, xbrl))

        if numero_esperado is not None and len(preguntas) != numero_esperado:
            problemas.append(
                f"hacen falta {numero_esperado} preguntas, hay {len(preguntas)}"
            )
        comparativas = sum(caso.familia == "comparativa" for caso in casos)
        if comparativas < minimo_comparativas:
            problemas.append(
                f"hacen falta {minimo_comparativas} comparativas, "
                f"hay {comparativas}"
            )
        return problemas

    @classmethod
    def cargar_jsonl(
        cls,
        ruta_jsonl: str | Path,
        corpus: CorpusVariant,
        *,
        numero_esperado: int | None = None,
        minimo_comparativas: int = 0,
    ) -> tuple["CasoGolden", ...]:
        """Carga cualquier JSONL que respete el contrato del golden set."""
        ruta = Path(ruta_jsonl)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el conjunto de evaluación: {ruta}")

        preguntas = []
        with ruta.open(encoding="utf-8") as fichero:
            for numero_linea, linea in enumerate(fichero, start=1):
                if not linea.strip():
                    continue
                try:
                    preguntas.append(json.loads(linea))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON inválido en {ruta}:{numero_linea}: {exc.msg}"
                    ) from exc

        secciones = pd.read_json(corpus.ruta_secciones, lines=True)
        xbrl = pd.read_parquet(corpus.ruta_xbrl)
        problemas = cls.validar_registros(
            preguntas,
            secciones=secciones,
            xbrl=xbrl,
            numero_esperado=numero_esperado,
            minimo_comparativas=minimo_comparativas,
        )
        if problemas:
            detalle = "\n".join(f"- {problema}" for problema in problemas)
            raise ValueError(f"El conjunto {ruta} no es válido:\n{detalle}")
        return tuple(cls.model_validate(pregunta) for pregunta in preguntas)
