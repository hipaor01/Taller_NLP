"""Descripción y validación de variantes del corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .chunking import FragmentoCorpus
from .hashing import calcular_sha256
from .indexing import ArtefactosIndice, calcular_huella_fragmentos


class CorpusVariant(BaseModel):
    """Conjunto coherente de fuentes, chunks e índice de una variante."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nombre: str = Field(min_length=1)
    ruta_secciones: Path
    ruta_xbrl: Path
    ruta_chunks: Path
    sha256_secciones: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    sha256_xbrl: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    sha256_chunks: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    troceador: str = Field(min_length=1)
    parametros_troceado: dict[str, JsonValue] = Field(default_factory=dict)
    ruta_indice_faiss: Path | None = None
    ruta_chunks_meta: Path | None = None
    modelo_embeddings: str | None = None
    dimension_embeddings: int | None = Field(default=None, gt=0)
    prefijo_consulta: str = ""
    artefactos_indice: ArtefactosIndice | None = None

    @field_validator("nombre", "troceador")
    @classmethod
    def normalizar_texto(cls, valor: str) -> str:
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator("modelo_embeddings")
    @classmethod
    def normalizar_modelo(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("modelo_embeddings no puede estar vacío.")
        return normalizado

    @field_validator("sha256_secciones", "sha256_xbrl", "sha256_chunks")
    @classmethod
    def normalizar_hash(cls, valor: str) -> str:
        return valor.lower()

    @model_validator(mode="after")
    def validar_artefactos(self) -> "CorpusVariant":
        archivos_con_hash = {
            self.ruta_secciones: self.sha256_secciones,
            self.ruta_xbrl: self.sha256_xbrl,
            self.ruta_chunks: self.sha256_chunks,
        }
        for ruta, hash_esperado in archivos_con_hash.items():
            self._exigir_fichero(ruta)
            hash_obtenido = calcular_sha256(ruta)
            if hash_obtenido != hash_esperado:
                raise ValueError(
                    f"El SHA-256 de {ruta} no coincide: "
                    f"esperado={hash_esperado}, obtenido={hash_obtenido}."
                )

        tiene_indice = self.ruta_indice_faiss is not None
        campos_indice = (
            self.ruta_chunks_meta,
            self.modelo_embeddings,
            self.dimension_embeddings,
        )
        if tiene_indice != all(valor is not None for valor in campos_indice):
            raise ValueError(
                "FAISS requiere ruta_chunks_meta, modelo_embeddings y "
                "dimension_embeddings; deben proporcionarse todos o ninguno."
            )
        if tiene_indice:
            self._validar_alineacion_faiss()
        if self.artefactos_indice is not None:
            self._validar_indice_reproducible()
        return self

    @staticmethod
    def _exigir_fichero(ruta: Path) -> None:
        if not ruta.is_file():
            raise ValueError(f"No existe el artefacto del corpus: {ruta}")

    @property
    def normalizar_embeddings(self) -> bool | None:
        """Indica cómo se generó el índice denso, cuando se conoce."""
        if self.artefactos_indice is not None:
            return self.artefactos_indice.normalizar_embeddings
        if self.modelo_embeddings is not None:
            return True
        return None

    def datos_manifiesto(self, directorio_base: Path) -> dict[str, Any]:
        """Serializa la variante con rutas relativas a ``directorio_base``."""
        if not isinstance(directorio_base, Path):
            raise TypeError("directorio_base debe ser pathlib.Path.")
        base = directorio_base.resolve()
        datos = self.model_dump(mode="json")

        def relativa(ruta: Path | None) -> str | None:
            if ruta is None:
                return None
            return os.path.relpath(ruta.resolve(), start=base)

        for campo in (
            "ruta_secciones",
            "ruta_xbrl",
            "ruta_chunks",
            "ruta_indice_faiss",
            "ruta_chunks_meta",
        ):
            datos[campo] = relativa(getattr(self, campo))

        if self.artefactos_indice is not None:
            datos["artefactos_indice"]["indice"]["ruta"] = relativa(
                self.artefactos_indice.indice.ruta
            )
            if self.artefactos_indice.metadatos is not None:
                datos["artefactos_indice"]["metadatos"]["ruta"] = relativa(
                    self.artefactos_indice.metadatos.ruta
                )
        return datos

    @classmethod
    def cargar_manifiesto(cls, ruta: Path) -> "CorpusVariant":
        """Carga un manifiesto resolviendo sus rutas desde su ubicación."""
        if not isinstance(ruta, Path):
            raise TypeError("ruta debe ser pathlib.Path.")
        if not ruta.is_file():
            raise ValueError(f"No existe el manifiesto del corpus: {ruta}")
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"El manifiesto no contiene JSON válido: {ruta}") from exc
        return cls.desde_datos_manifiesto(
            datos,
            directorio_base=ruta.resolve().parent,
        )

    @classmethod
    def desde_datos_manifiesto(
        cls,
        datos: dict[str, Any],
        *,
        directorio_base: Path,
    ) -> "CorpusVariant":
        """Reconstruye una variante desde datos con rutas relativas."""
        if not isinstance(datos, dict):
            raise TypeError("datos debe ser un diccionario.")
        if not isinstance(directorio_base, Path):
            raise TypeError("directorio_base debe ser pathlib.Path.")
        # La copia profunda impide modificar el diccionario recibido al
        # resolver también las rutas anidadas de los artefactos del índice.
        datos = json.loads(json.dumps(datos, ensure_ascii=False))
        base = directorio_base.resolve()

        def absoluta(valor: str | None) -> str | None:
            if valor is None:
                return None
            candidata = Path(valor)
            return str(
                candidata.resolve()
                if candidata.is_absolute()
                else (base / candidata).resolve()
            )

        for campo in (
            "ruta_secciones",
            "ruta_xbrl",
            "ruta_chunks",
            "ruta_indice_faiss",
            "ruta_chunks_meta",
        ):
            datos[campo] = absoluta(datos.get(campo))

        artefactos = datos.get("artefactos_indice")
        if artefactos is not None:
            artefactos["indice"]["ruta"] = absoluta(
                artefactos["indice"]["ruta"]
            )
            if artefactos.get("metadatos") is not None:
                artefactos["metadatos"]["ruta"] = absoluta(
                    artefactos["metadatos"]["ruta"]
                )
        return cls.model_validate(datos)

    def _validar_alineacion_faiss(self) -> None:
        assert self.ruta_indice_faiss is not None
        assert self.ruta_chunks_meta is not None
        assert self.dimension_embeddings is not None

        self._exigir_fichero(self.ruta_indice_faiss)
        self._exigir_fichero(self.ruta_chunks_meta)

        fragmentos = self._cargar_fragmentos()
        ids_chunks = [fragmento.chunk_id for fragmento in fragmentos]

        if len(ids_chunks) != len(set(ids_chunks)):
            raise ValueError("ruta_chunks contiene chunk_id duplicados.")

        metadatos = pd.read_parquet(
            self.ruta_chunks_meta, columns=["chunk_id"]
        )
        ids_metadatos = metadatos["chunk_id"].tolist()
        if ids_metadatos != ids_chunks:
            raise ValueError(
                "El orden de chunk_id no coincide entre chunks y metadatos."
            )

        indice = faiss.read_index(str(self.ruta_indice_faiss))
        if indice.ntotal != len(ids_chunks):
            raise ValueError(
                f"FAISS contiene {indice.ntotal} vectores y hay "
                f"{len(ids_chunks)} chunks."
            )
        if indice.d != self.dimension_embeddings:
            raise ValueError(
                f"FAISS tiene dimensión {indice.d}, no "
                f"{self.dimension_embeddings}."
            )

    def _validar_indice_reproducible(self) -> None:
        assert self.artefactos_indice is not None
        artefactos = self.artefactos_indice
        fragmentos = self._cargar_fragmentos()
        if artefactos.numero_fragmentos != len(fragmentos):
            raise ValueError(
                f"El índice declara {artefactos.numero_fragmentos} fragmentos "
                f"y el corpus contiene {len(fragmentos)}."
            )
        huella = calcular_huella_fragmentos(fragmentos)
        if artefactos.sha256_fragmentos != huella:
            raise ValueError(
                "La huella de los fragmentos no coincide con la usada para "
                "construir el índice."
            )

        if artefactos.formato != "faiss":
            return
        if self.ruta_indice_faiss is None or self.ruta_chunks_meta is None:
            raise ValueError(
                "Un ArtefactosIndice FAISS debe exponerse también mediante "
                "ruta_indice_faiss y ruta_chunks_meta."
            )
        if artefactos.metadatos is None:
            raise ValueError("Los artefactos FAISS requieren metadatos.")
        coherencias = (
            artefactos.indice.ruta.resolve()
            == self.ruta_indice_faiss.resolve(),
            artefactos.metadatos.ruta.resolve()
            == self.ruta_chunks_meta.resolve(),
            artefactos.modelo_embeddings == self.modelo_embeddings,
            artefactos.dimension_embeddings == self.dimension_embeddings,
            artefactos.prefijo_consulta == self.prefijo_consulta,
        )
        if not all(coherencias):
            raise ValueError(
                "ArtefactosIndice y los campos FAISS de CorpusVariant no "
                "describen el mismo índice."
            )

    def _cargar_fragmentos(self) -> tuple[FragmentoCorpus, ...]:
        fragmentos = []
        with self.ruta_chunks.open(encoding="utf-8") as fichero:
            for numero_linea, linea in enumerate(fichero, start=1):
                if not linea.strip():
                    continue
                try:
                    fragmentos.append(FragmentoCorpus.model_validate_json(linea))
                except ValueError as exc:
                    raise ValueError(
                        f"Chunk inválido en {self.ruta_chunks}:{numero_linea}."
                    ) from exc
        return tuple(fragmentos)
