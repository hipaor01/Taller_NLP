"""Manifiestos portables para registrar experimentos reproducibles."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import textwrap
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .assembly import ConstructorAgente
from .config import ConfiguracionAgente
from .contracts import InformeEvaluacion, NombreHerramienta
from .corpus import CorpusVariant
from .hashing import calcular_sha256


_DISTRIBUCIONES = (
    "taller-nlp",
    "langchain",
    "langchain-core",
    "langgraph",
    "langchain-openrouter",
    "langchain-huggingface",
    "sentence-transformers",
    "faiss-cpu",
    "rank-bm25",
    "pandas",
    "pyarrow",
    "pydantic",
)
_NOMBRES_TOOLS = frozenset(
    {"list_available", "get_xbrl_fact", "search_filings", "read_section"}
)
_CLAVES_SENSIBLES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "access_token",
        "refresh_token",
        "openrouter_api_key",
    }
)


def _exigir_ausencia_de_secretos(
    valor: JsonValue,
    *,
    ruta: str,
) -> JsonValue:
    if isinstance(valor, dict):
        for clave, contenido in valor.items():
            clave_normalizada = clave.strip().casefold()
            if clave_normalizada in _CLAVES_SENSIBLES:
                raise ValueError(
                    f"{ruta}.{clave} parece contener una credencial y no "
                    "puede guardarse en el manifiesto."
                )
            _exigir_ausencia_de_secretos(
                contenido,
                ruta=f"{ruta}.{clave}",
            )
    elif isinstance(valor, list):
        for indice, contenido in enumerate(valor):
            _exigir_ausencia_de_secretos(
                contenido,
                ruta=f"{ruta}[{indice}]",
            )
    return valor


class DescriptorComponente(BaseModel):
    """Identidad importable y parámetros declarados de un componente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    referencia: str = Field(min_length=1)
    nombre: str | None = None
    parametros: dict[str, JsonValue] = Field(default_factory=dict)
    sha256_fuente: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )

    @field_validator("referencia", "nombre")
    @classmethod
    def normalizar_texto(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator("sha256_fuente")
    @classmethod
    def normalizar_hash(cls, valor: str | None) -> str | None:
        return valor.lower() if valor is not None else None

    @field_validator("parametros")
    @classmethod
    def rechazar_secretos(
        cls, valor: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _exigir_ausencia_de_secretos(valor, ruta="parametros")
        return valor


@final
class ManifiestoExperimento(BaseModel):
    """Instantánea portable de una configuración y su evaluación opcional.

    El manifiesto contiene únicamente configuración no sensible. En
    particular, nunca inspecciona variables de entorno ni credenciales de los
    modelos. Las rutas del corpus y del golden set se vuelven relativas al
    fichero al persistirlo.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version_esquema: Literal[2] = 2
    nombre_experimento: str = Field(min_length=1)
    creado_utc: datetime
    revision_codigo: str | None = None
    sha256_codigo: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    version_python: str = Field(min_length=1)
    plataforma: str = Field(min_length=1)
    versiones_dependencias: dict[str, str] = Field(min_length=1)

    configuracion_agente: ConfiguracionAgente
    corpus: CorpusVariant
    sha256_indice: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    sha256_metadatos_indice: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    implementaciones_herramientas: dict[
        NombreHerramienta, DescriptorComponente
    ]
    descripciones_herramientas: dict[NombreHerramienta, str]
    retriever: DescriptorComponente | None = None
    middlewares_usuario: tuple[DescriptorComponente, ...] = ()
    modelo_inyectado: DescriptorComponente | None = None

    k_retrieval: int = Field(gt=0)
    tolerancia_absoluta: float = Field(ge=0)
    tolerancia_relativa: float = Field(ge=0)
    numero_esperado: int | None = Field(default=None, gt=0)
    minimo_comparativas: int = Field(default=0, ge=0)

    informe: InformeEvaluacion | None = None
    sha256_golden: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    metadatos: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("nombre_experimento", "revision_codigo")
    @classmethod
    def normalizar_texto(cls, valor: str | None) -> str | None:
        if valor is None:
            return None
        normalizado = valor.strip()
        if not normalizado:
            raise ValueError("El texto no puede estar vacío.")
        return normalizado

    @field_validator(
        "sha256_codigo",
        "sha256_golden",
        "sha256_indice",
        "sha256_metadatos_indice",
    )
    @classmethod
    def normalizar_hash(cls, valor: str | None) -> str | None:
        return valor.lower() if valor is not None else None

    @field_validator("creado_utc")
    @classmethod
    def exigir_zona_horaria(cls, valor: datetime) -> datetime:
        if valor.tzinfo is None or valor.utcoffset() is None:
            raise ValueError("creado_utc debe incluir zona horaria.")
        return valor

    @field_validator("metadatos")
    @classmethod
    def rechazar_secretos_metadatos(
        cls, valor: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _exigir_ausencia_de_secretos(valor, ruta="metadatos")
        return valor

    @model_validator(mode="after")
    def validar_coherencia(self) -> "ManifiestoExperimento":
        if set(self.implementaciones_herramientas) != _NOMBRES_TOOLS:
            raise ValueError(
                "El manifiesto debe identificar exactamente las cuatro tools."
            )
        if set(self.descripciones_herramientas) != _NOMBRES_TOOLS:
            raise ValueError(
                "El manifiesto debe contener las cuatro descripciones de tools."
            )
        tiene_faiss = self.corpus.ruta_indice_faiss is not None
        tiene_hashes_faiss = (
            self.sha256_indice is not None
            and self.sha256_metadatos_indice is not None
        )
        if tiene_faiss != tiene_hashes_faiss:
            raise ValueError(
                "Un corpus FAISS requiere los hashes de índice y metadatos."
            )
        if (self.informe is None) != (self.sha256_golden is None):
            raise ValueError(
                "informe y sha256_golden deben aparecer juntos."
            )
        if self.informe is not None:
            if self.informe.nombre_agente != self.nombre_experimento:
                raise ValueError(
                    "El informe pertenece a un agente distinto del experimento."
                )
            if (
                self.informe.k_retrieval != self.k_retrieval
                or self.informe.tolerancia_absoluta
                != self.tolerancia_absoluta
                or self.informe.tolerancia_relativa
                != self.tolerancia_relativa
            ):
                raise ValueError(
                    "El informe no usa la configuración de evaluación del "
                    "experimento."
                )
        return self

    @classmethod
    def desde_constructor(
        cls,
        constructor: ConstructorAgente,
        *,
        informe: InformeEvaluacion | None = None,
        revision_codigo: str | None = None,
        metadatos: Mapping[str, JsonValue] | None = None,
    ) -> "ManifiestoExperimento":
        """Captura el ensamblado antes o después de evaluarlo."""
        if not isinstance(constructor, ConstructorAgente):
            raise TypeError("constructor debe ser un ConstructorAgente.")
        if informe is not None and not isinstance(informe, InformeEvaluacion):
            raise TypeError("informe debe ser un InformeEvaluacion o None.")

        fabrica = constructor.fabrica_herramientas
        implementaciones = {
            nombre: cls._describir(
                implementacion,
                nombre=nombre,
            )
            for nombre, implementacion in fabrica.implementaciones.items()
        }
        retriever = fabrica.retriever
        evaluador = constructor.evaluador
        sha256_indice = (
            calcular_sha256(constructor.corpus.ruta_indice_faiss)
            if constructor.corpus.ruta_indice_faiss is not None
            else None
        )
        sha256_metadatos_indice = (
            calcular_sha256(constructor.corpus.ruta_chunks_meta)
            if constructor.corpus.ruta_chunks_meta is not None
            else None
        )
        sha256_golden = None
        if informe is not None:
            ruta_golden = Path(informe.ruta_jsonl)
            if not ruta_golden.is_file():
                raise ValueError(
                    f"No existe el golden set del informe: {ruta_golden}"
                )
            sha256_golden = calcular_sha256(ruta_golden)

        return cls(
            nombre_experimento=constructor.nombre,
            creado_utc=datetime.now(timezone.utc),
            revision_codigo=revision_codigo,
            sha256_codigo=cls._calcular_sha256_codigo(),
            version_python=platform.python_version(),
            plataforma=platform.platform(),
            versiones_dependencias=cls._obtener_versiones(),
            configuracion_agente=constructor.configuracion,
            corpus=constructor.corpus,
            sha256_indice=sha256_indice,
            sha256_metadatos_indice=sha256_metadatos_indice,
            implementaciones_herramientas=implementaciones,
            descripciones_herramientas=dict(fabrica.descripciones),
            retriever=(
                cls._describir(
                    retriever,
                    nombre=retriever.nombre,
                    parametros=dict(retriever.parametros),
                )
                if retriever is not None
                else None
            ),
            middlewares_usuario=tuple(
                cls._describir(middleware)
                for middleware in constructor.middlewares
            ),
            modelo_inyectado=(
                cls._describir(constructor.modelo)
                if constructor.modelo is not None
                else None
            ),
            k_retrieval=evaluador.k_retrieval,
            tolerancia_absoluta=evaluador.tolerancia_absoluta,
            tolerancia_relativa=evaluador.tolerancia_relativa,
            numero_esperado=evaluador.numero_esperado,
            minimo_comparativas=evaluador.minimo_comparativas,
            informe=informe,
            sha256_golden=sha256_golden,
            metadatos=dict(metadatos or {}),
        )

    def guardar(self, ruta: str | Path) -> Path:
        """Guarda el manifiesto atómicamente sin sobrescribir otro."""
        destino = Path(ruta).resolve()
        if destino.exists():
            raise FileExistsError(
                f"El manifiesto ya existe y no se sobrescribirá: {destino}"
            )
        if not destino.parent.is_dir():
            raise ValueError(
                f"El directorio padre no existe: {destino.parent}"
            )

        datos = self.model_dump(mode="json", exclude_computed_fields=True)
        datos["corpus"] = self.corpus.datos_manifiesto(destino.parent)
        if self.informe is not None:
            ruta_golden = Path(self.informe.ruta_jsonl).resolve()
            datos["informe"]["ruta_jsonl"] = os.path.relpath(
                ruta_golden,
                start=destino.parent,
            )

        with TemporaryDirectory(
            prefix=".manifiesto-experimento-",
            dir=destino.parent,
        ) as temporal:
            provisional = Path(temporal) / destino.name
            provisional.write_text(
                json.dumps(datos, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            provisional.rename(destino)
        return destino

    @classmethod
    def cargar(cls, ruta: str | Path) -> "ManifiestoExperimento":
        """Carga el manifiesto y verifica corpus y golden set referenciados."""
        origen = Path(ruta)
        if not origen.is_file():
            raise FileNotFoundError(f"No existe el manifiesto: {origen}")
        try:
            datos = json.loads(origen.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"El manifiesto no contiene JSON válido: {origen}"
            ) from exc
        if not isinstance(datos, dict):
            raise ValueError("La raíz del manifiesto debe ser un objeto JSON.")

        base = origen.resolve().parent
        datos["corpus"] = CorpusVariant.desde_datos_manifiesto(
            datos.get("corpus"),
            directorio_base=base,
        )
        informe = datos.get("informe")
        if informe is not None:
            ruta_golden = Path(informe["ruta_jsonl"])
            if not ruta_golden.is_absolute():
                ruta_golden = (base / ruta_golden).resolve()
            informe["ruta_jsonl"] = str(ruta_golden)

        manifiesto = cls.model_validate(datos)
        if manifiesto.informe is not None:
            ruta_golden = Path(manifiesto.informe.ruta_jsonl)
            if not ruta_golden.is_file():
                raise ValueError(
                    f"No existe el golden set del manifiesto: {ruta_golden}"
                )
            obtenido = calcular_sha256(ruta_golden)
            if obtenido != manifiesto.sha256_golden:
                raise ValueError(
                    "El SHA-256 del golden set no coincide con el manifiesto."
                )
        manifiesto._verificar_hash_faiss(
            manifiesto.corpus.ruta_indice_faiss,
            manifiesto.sha256_indice,
            etiqueta="índice FAISS",
        )
        manifiesto._verificar_hash_faiss(
            manifiesto.corpus.ruta_chunks_meta,
            manifiesto.sha256_metadatos_indice,
            etiqueta="metadatos FAISS",
        )
        return manifiesto

    def comprobar_reproducibilidad(self) -> tuple[str, ...]:
        """Enumera diferencias entre el manifiesto y el estado actual."""
        diferencias = []
        if self._calcular_sha256_codigo() != self.sha256_codigo:
            diferencias.append("La huella del código de taller_nlp ha cambiado.")
        if platform.python_version() != self.version_python:
            diferencias.append(
                f"Python es {platform.python_version()}, no "
                f"{self.version_python}."
            )
        if platform.platform() != self.plataforma:
            diferencias.append("La plataforma de ejecución ha cambiado.")

        actuales = self._obtener_versiones()
        for distribucion, esperada in self.versiones_dependencias.items():
            obtenida = actuales.get(distribucion, "no-disponible")
            if obtenida != esperada:
                diferencias.append(
                    f"{distribucion} tiene versión {obtenida}, no {esperada}."
                )

        artefactos: list[tuple[str, Path, str]] = [
            (
                "secciones",
                self.corpus.ruta_secciones,
                self.corpus.sha256_secciones,
            ),
            ("XBRL", self.corpus.ruta_xbrl, self.corpus.sha256_xbrl),
            ("chunks", self.corpus.ruta_chunks, self.corpus.sha256_chunks),
        ]
        if self.corpus.ruta_indice_faiss is not None:
            assert self.sha256_indice is not None
            artefactos.append(
                (
                    "índice FAISS",
                    self.corpus.ruta_indice_faiss,
                    self.sha256_indice,
                )
            )
        if self.corpus.ruta_chunks_meta is not None:
            assert self.sha256_metadatos_indice is not None
            artefactos.append(
                (
                    "metadatos FAISS",
                    self.corpus.ruta_chunks_meta,
                    self.sha256_metadatos_indice,
                )
            )
        if self.corpus.artefactos_indice is not None:
            indice = self.corpus.artefactos_indice
            artefactos.append(
                ("índice", indice.indice.ruta, indice.indice.sha256)
            )
            if indice.metadatos is not None:
                artefactos.append(
                    (
                        "metadatos del índice",
                        indice.metadatos.ruta,
                        indice.metadatos.sha256,
                    )
                )

        vistos: set[Path] = set()
        for etiqueta, ruta, esperado in artefactos:
            ruta_resuelta = ruta.resolve()
            if ruta_resuelta in vistos:
                continue
            vistos.add(ruta_resuelta)
            if not ruta.is_file():
                diferencias.append(f"No existe el artefacto {etiqueta}: {ruta}.")
            elif calcular_sha256(ruta) != esperado:
                diferencias.append(f"El artefacto {etiqueta} ha cambiado.")

        if self.informe is not None:
            ruta_golden = Path(self.informe.ruta_jsonl)
            if not ruta_golden.is_file():
                diferencias.append(f"No existe el golden set: {ruta_golden}.")
            elif calcular_sha256(ruta_golden) != self.sha256_golden:
                diferencias.append("El golden set ha cambiado.")
        return tuple(diferencias)

    @staticmethod
    def _verificar_hash_faiss(
        ruta: Path | None,
        esperado: str | None,
        *,
        etiqueta: str,
    ) -> None:
        if ruta is None:
            return
        assert esperado is not None
        if calcular_sha256(ruta) != esperado:
            raise ValueError(
                f"El SHA-256 de {etiqueta} no coincide con el manifiesto."
            )

    @staticmethod
    def _describir(
        componente: Any,
        *,
        nombre: str | None = None,
        parametros: Mapping[str, JsonValue] | None = None,
    ) -> DescriptorComponente:
        objetivo = componente.__func__ if inspect.ismethod(componente) else componente
        if inspect.isfunction(objetivo) or inspect.isclass(objetivo):
            referencia = f"{objetivo.__module__}.{objetivo.__qualname__}"
            fuente_objetivo = objetivo
        else:
            referencia = (
                f"{type(componente).__module__}."
                f"{type(componente).__qualname__}"
            )
            fuente_objetivo = type(componente)

        if parametros is None:
            parametros_declarados = getattr(componente, "parametros", {})
            parametros = (
                dict(parametros_declarados)
                if isinstance(parametros_declarados, Mapping)
                else {}
            )
        try:
            fuente = textwrap.dedent(inspect.getsource(fuente_objetivo)).strip()
        except (OSError, TypeError):
            sha256_fuente = None
        else:
            sha256_fuente = hashlib.sha256(
                fuente.encode("utf-8")
            ).hexdigest()
        return DescriptorComponente(
            referencia=referencia,
            nombre=nombre,
            parametros=dict(parametros),
            sha256_fuente=sha256_fuente,
        )

    @staticmethod
    def _calcular_sha256_codigo() -> str:
        raiz = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for ruta in sorted(raiz.rglob("*.py")):
            digest.update(ruta.relative_to(raiz).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(ruta.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _obtener_versiones() -> dict[str, str]:
        versiones = {}
        for distribucion in _DISTRIBUCIONES:
            try:
                versiones[distribucion] = version(distribucion)
            except PackageNotFoundError:
                versiones[distribucion] = "no-disponible"
        return versiones
