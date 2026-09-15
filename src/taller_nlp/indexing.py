"""Contratos para construir índices reproducibles sobre el corpus."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .chunking import FragmentoCorpus
from .component import ComponenteConfigurable
from .hashing import calcular_sha256


def calcular_huella_fragmentos(
    fragmentos: Sequence[FragmentoCorpus],
) -> str:
    """Calcula una huella sensible al contenido y al orden de los chunks."""
    digest = hashlib.sha256()
    for fragmento in fragmentos:
        digest.update(fragmento.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class ArchivoArtefacto(BaseModel):
    """Fichero producido durante la indexación y ligado a su hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ruta: Path
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    @field_validator("sha256")
    @classmethod
    def normalizar_hash(cls, valor: str) -> str:
        return valor.lower()

    @model_validator(mode="after")
    def validar_fichero(self) -> "ArchivoArtefacto":
        if not self.ruta.is_file():
            raise ValueError(f"No existe el artefacto de índice: {self.ruta}")
        obtenido = calcular_sha256(self.ruta)
        if obtenido != self.sha256:
            raise ValueError(
                f"El SHA-256 de {self.ruta} no coincide: "
                f"esperado={self.sha256}, obtenido={obtenido}."
            )
        return self

    @classmethod
    def desde_ruta(cls, ruta: Path) -> "ArchivoArtefacto":
        if not ruta.is_file():
            raise ValueError(f"No existe el artefacto de índice: {ruta}")
        ruta_resuelta = ruta.resolve()
        return cls(
            ruta=ruta_resuelta,
            sha256=calcular_sha256(ruta_resuelta),
        )


class ArtefactosIndice(BaseModel):
    """Resultado reproducible y agnóstico al formato de una indexación."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nombre_indexador: str = Field(min_length=1)
    formato: str = Field(min_length=1)
    indice: ArchivoArtefacto
    metadatos: ArchivoArtefacto | None = None
    numero_fragmentos: int = Field(gt=0)
    sha256_fragmentos: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    parametros_indexacion: dict[str, JsonValue] = Field(default_factory=dict)
    modelo_embeddings: str | None = None
    dimension_embeddings: int | None = Field(default=None, gt=0)
    normalizar_embeddings: bool | None = None
    prefijo_consulta: str = ""

    @field_validator("nombre_indexador", "formato", "modelo_embeddings")
    @classmethod
    def normalizar_texto(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator("formato")
    @classmethod
    def normalizar_formato(cls, valor: str) -> str:
        return valor.lower()

    @field_validator("sha256_fragmentos")
    @classmethod
    def normalizar_hash(cls, valor: str) -> str:
        return valor.lower()

    @model_validator(mode="after")
    def validar_configuracion_embeddings(self) -> "ArtefactosIndice":
        configuracion = (
            self.modelo_embeddings,
            self.dimension_embeddings,
            self.normalizar_embeddings,
        )
        if any(valor is not None for valor in configuracion) and not all(
            valor is not None for valor in configuracion
        ):
            raise ValueError(
                "modelo_embeddings, dimension_embeddings y "
                "normalizar_embeddings deben indicarse juntos."
            )
        if self.modelo_embeddings is None and self.prefijo_consulta:
            raise ValueError(
                "prefijo_consulta solo puede usarse con un modelo de embeddings."
            )
        return self


class IndexadorCorpus(ComponenteConfigurable, ABC):
    """Base para indexadores densos, léxicos o de otro formato.

    La subclase implementa ``_construir_indice`` y conserva el orden recibido
    de los fragmentos. La clase base valida la entrada, liga los artefactos a
    hashes y devuelve una descripción reproducible de la indexación.
    """

    def __init__(
        self,
        nombre: str,
        formato: str,
        *,
        parametros: Mapping[str, JsonValue] | None = None,
        modelo_embeddings: str | None = None,
        dimension_embeddings: int | None = None,
        normalizar_embeddings: bool | None = None,
        prefijo_consulta: str = "",
    ) -> None:
        super().__init__(nombre, parametros=parametros)
        if not isinstance(formato, str) or not formato.strip():
            raise ValueError("formato debe ser texto no vacío.")
        if modelo_embeddings is not None and (
            not isinstance(modelo_embeddings, str)
            or not modelo_embeddings.strip()
        ):
            raise ValueError("modelo_embeddings debe ser texto no vacío o None.")
        if dimension_embeddings is not None and (
            isinstance(dimension_embeddings, bool)
            or not isinstance(dimension_embeddings, int)
            or dimension_embeddings <= 0
        ):
            raise ValueError(
                "dimension_embeddings debe ser un entero mayor que cero o None."
            )
        if normalizar_embeddings is not None and not isinstance(
            normalizar_embeddings, bool
        ):
            raise TypeError("normalizar_embeddings debe ser bool o None.")
        if not isinstance(prefijo_consulta, str):
            raise TypeError("prefijo_consulta debe ser texto.")

        configuracion_embeddings = (
            modelo_embeddings,
            dimension_embeddings,
            normalizar_embeddings,
        )
        if any(valor is not None for valor in configuracion_embeddings) and not all(
            valor is not None for valor in configuracion_embeddings
        ):
            raise ValueError(
                "modelo_embeddings, dimension_embeddings y "
                "normalizar_embeddings deben indicarse juntos."
            )
        if modelo_embeddings is None and prefijo_consulta:
            raise ValueError(
                "prefijo_consulta solo puede usarse con un modelo de embeddings."
            )

        self._formato = formato.strip().lower()
        self._modelo_embeddings = (
            modelo_embeddings.strip() if modelo_embeddings is not None else None
        )
        self._dimension_embeddings = dimension_embeddings
        self._normalizar_embeddings = normalizar_embeddings
        self._prefijo_consulta = prefijo_consulta

    @property
    def formato(self) -> str:
        return self._formato

    @property
    def modelo_embeddings(self) -> str | None:
        return self._modelo_embeddings

    @property
    def dimension_embeddings(self) -> int | None:
        return self._dimension_embeddings

    @property
    def normalizar_embeddings(self) -> bool | None:
        return self._normalizar_embeddings

    @property
    def prefijo_consulta(self) -> str:
        return self._prefijo_consulta

    def indexar(
        self,
        fragmentos: Sequence[FragmentoCorpus],
        directorio_salida: Path,
    ) -> ArtefactosIndice:
        """Construye y valida los artefactos para una secuencia de chunks."""
        fragmentos = tuple(fragmentos)
        self._validar_fragmentos(fragmentos)
        if not isinstance(directorio_salida, Path):
            raise TypeError("directorio_salida debe ser pathlib.Path.")
        if not directorio_salida.is_dir():
            raise ValueError(
                f"El directorio de salida no existe: {directorio_salida}"
            )

        salida = self._construir_indice(fragmentos, directorio_salida)
        if (
            not isinstance(salida, tuple)
            or len(salida) != 2
            or not isinstance(salida[0], Path)
            or (salida[1] is not None and not isinstance(salida[1], Path))
        ):
            raise TypeError(
                "_construir_indice debe devolver tuple[Path, Path | None]."
            )
        ruta_indice, ruta_metadatos = salida
        self._validar_ruta_salida(ruta_indice, directorio_salida)
        if ruta_metadatos is not None:
            self._validar_ruta_salida(ruta_metadatos, directorio_salida)
            if ruta_metadatos.resolve() == ruta_indice.resolve():
                raise ValueError(
                    "El índice y sus metadatos deben ser ficheros distintos."
                )

        return ArtefactosIndice(
            nombre_indexador=self.nombre,
            formato=self._formato,
            indice=ArchivoArtefacto.desde_ruta(ruta_indice),
            metadatos=(
                ArchivoArtefacto.desde_ruta(ruta_metadatos)
                if ruta_metadatos is not None
                else None
            ),
            numero_fragmentos=len(fragmentos),
            sha256_fragmentos=calcular_huella_fragmentos(fragmentos),
            parametros_indexacion=dict(self.parametros),
            modelo_embeddings=self._modelo_embeddings,
            dimension_embeddings=self._dimension_embeddings,
            normalizar_embeddings=self._normalizar_embeddings,
            prefijo_consulta=self._prefijo_consulta,
        )

    @abstractmethod
    def _construir_indice(
        self,
        fragmentos: tuple[FragmentoCorpus, ...],
        directorio_salida: Path,
    ) -> tuple[Path, Path | None]:
        """Escribe el índice y, opcionalmente, sus metadatos alineados."""

    @staticmethod
    def _validar_fragmentos(
        fragmentos: tuple[FragmentoCorpus, ...],
    ) -> None:
        if not fragmentos:
            raise ValueError("No se puede indexar una secuencia vacía.")
        if any(
            not isinstance(fragmento, FragmentoCorpus)
            for fragmento in fragmentos
        ):
            raise TypeError("Solo pueden indexarse objetos FragmentoCorpus.")

        ids = [fragmento.chunk_id for fragmento in fragmentos]
        if len(ids) != len(set(ids)):
            raise ValueError("No pueden indexarse chunk_id duplicados.")

        posiciones_por_seccion: dict[tuple[str, int, str], list[int]] = (
            defaultdict(list)
        )
        for fragmento in fragmentos:
            clave = (
                fragmento.ticker,
                fragmento.fiscal_year,
                fragmento.item,
            )
            posiciones_por_seccion[clave].append(fragmento.posicion)
        for clave, posiciones in posiciones_por_seccion.items():
            if posiciones != list(range(len(posiciones))):
                raise ValueError(
                    f"Las posiciones de la sección {clave} no son "
                    "contiguas o no conservan su orden."
                )

    @staticmethod
    def _validar_ruta_salida(ruta: Path, directorio_salida: Path) -> None:
        if not ruta.resolve().is_relative_to(directorio_salida.resolve()):
            raise ValueError(
                f"El artefacto {ruta} está fuera del directorio de salida."
            )
