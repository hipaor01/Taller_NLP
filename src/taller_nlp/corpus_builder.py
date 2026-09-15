"""Orquestación reproducible de nuevas variantes del corpus."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .chunking import FragmentoCorpus, TroceadorCorpus
from .corpus import CorpusVariant
from .hashing import calcular_sha256
from .indexing import (
    ArtefactosIndice,
    IndexadorCorpus,
)


_COLUMNAS_SECCIONES = frozenset({"ticker", "fiscal_year", "item", "texto"})


class ConstructorCorpusVariant:
    """Construye una variante mediante componentes inyectados.

    La construcción se realiza en un directorio temporal. El destino final no
    se publica hasta que chunks, índice y ``CorpusVariant`` han superado todas
    sus validaciones.
    """

    def __init__(
        self,
        corpus_fuente: CorpusVariant,
        troceador: TroceadorCorpus,
        *,
        indexador: IndexadorCorpus | None = None,
    ) -> None:
        if not isinstance(corpus_fuente, CorpusVariant):
            raise TypeError("corpus_fuente debe ser una CorpusVariant.")
        if not isinstance(troceador, TroceadorCorpus):
            raise TypeError("troceador debe ser una instancia de TroceadorCorpus.")
        if indexador is not None and not isinstance(indexador, IndexadorCorpus):
            raise TypeError("indexador debe ser una instancia de IndexadorCorpus.")
        self._corpus_fuente = corpus_fuente
        self._troceador = troceador
        self._indexador = indexador

    @property
    def corpus_fuente(self) -> CorpusVariant:
        return self._corpus_fuente

    @property
    def troceador(self) -> TroceadorCorpus:
        return self._troceador

    @property
    def indexador(self) -> IndexadorCorpus | None:
        return self._indexador

    def construir(
        self,
        nombre: str,
        directorio_salida: Path,
    ) -> CorpusVariant:
        """Construye la variante sin sobrescribir un destino existente."""
        nombre = self._normalizar_nombre(nombre)
        if not isinstance(directorio_salida, Path):
            raise TypeError("directorio_salida debe ser pathlib.Path.")
        destino = directorio_salida.resolve()
        if destino.exists():
            raise FileExistsError(
                f"El destino ya existe y no se sobrescribirá: {destino}"
            )
        if not destino.parent.is_dir():
            raise ValueError(
                f"El directorio padre no existe: {destino.parent}"
            )

        fragmentos = self._trocear_secciones()
        with TemporaryDirectory(
            prefix=".construccion-corpus-",
            dir=destino.parent,
        ) as temporal:
            staging = Path(temporal).resolve()
            ruta_chunks = staging / "chunks.jsonl"
            self._escribir_chunks(fragmentos, ruta_chunks)

            artefactos = self._construir_indice(fragmentos, staging)
            variante_temporal = self._crear_variante(
                nombre=nombre,
                ruta_chunks=ruta_chunks,
                artefactos=artefactos,
            )
            variante_final = self._remapear_variante(
                variante_temporal,
                origen=staging,
                destino=destino,
            )
            self._escribir_manifiesto(
                variante_final,
                staging / "corpus_variant.json",
                directorio_base=destino,
            )
            staging.rename(destino)
            return variante_final

    def _trocear_secciones(self) -> tuple[FragmentoCorpus, ...]:
        secciones = pd.read_json(
            self._corpus_fuente.ruta_secciones,
            lines=True,
        )
        ausentes = _COLUMNAS_SECCIONES - frozenset(secciones.columns)
        if ausentes:
            raise ValueError(
                f"Faltan columnas en secciones.jsonl: {sorted(ausentes)}."
            )
        if secciones[list(_COLUMNAS_SECCIONES)].isna().any().any():
            raise ValueError("secciones.jsonl contiene campos requeridos nulos.")
        claves = ["ticker", "fiscal_year", "item"]
        if secciones.duplicated(claves).any():
            raise ValueError(
                "secciones.jsonl contiene ticker, fiscal_year e item duplicados."
            )

        fragmentos = []
        for fila in secciones.itertuples(index=False):
            fragmentos.extend(
                self._troceador.trocear(
                    ticker=str(fila.ticker),
                    fiscal_year=int(fila.fiscal_year),
                    item=str(fila.item),
                    texto=str(fila.texto),
                )
            )
        ids = [fragmento.chunk_id for fragmento in fragmentos]
        if len(ids) != len(set(ids)):
            raise ValueError("El troceado completo produjo chunk_id duplicados.")
        return tuple(fragmentos)

    @staticmethod
    def _escribir_chunks(
        fragmentos: tuple[FragmentoCorpus, ...],
        ruta: Path,
    ) -> None:
        with ruta.open("x", encoding="utf-8", newline="\n") as fichero:
            for fragmento in fragmentos:
                fichero.write(fragmento.model_dump_json())
                fichero.write("\n")

    def _construir_indice(
        self,
        fragmentos: tuple[FragmentoCorpus, ...],
        staging: Path,
    ) -> ArtefactosIndice | None:
        if self._indexador is None:
            return None
        directorio_indice = staging / "indice"
        directorio_indice.mkdir()
        return self._indexador.indexar(fragmentos, directorio_indice)

    @staticmethod
    def _escribir_manifiesto(
        variante: CorpusVariant,
        ruta: Path,
        *,
        directorio_base: Path,
    ) -> None:
        with ruta.open("x", encoding="utf-8", newline="\n") as fichero:
            json.dump(
                variante.datos_manifiesto(directorio_base),
                fichero,
                ensure_ascii=False,
                indent=2,
            )
            fichero.write("\n")

    def _crear_variante(
        self,
        *,
        nombre: str,
        ruta_chunks: Path,
        artefactos: ArtefactosIndice | None,
    ) -> CorpusVariant:
        campos_faiss: dict[str, object] = {}
        if artefactos is not None and artefactos.formato == "faiss":
            if (
                artefactos.metadatos is None
                or artefactos.modelo_embeddings is None
                or artefactos.dimension_embeddings is None
                or artefactos.normalizar_embeddings is None
            ):
                raise ValueError(
                    "Un indexador FAISS debe producir metadatos y declarar "
                    "modelo, dimensión y normalización de embeddings."
                )
            campos_faiss = {
                "ruta_indice_faiss": artefactos.indice.ruta,
                "ruta_chunks_meta": artefactos.metadatos.ruta,
                "modelo_embeddings": artefactos.modelo_embeddings,
                "dimension_embeddings": artefactos.dimension_embeddings,
                "prefijo_consulta": artefactos.prefijo_consulta,
            }

        return CorpusVariant(
            nombre=nombre,
            ruta_secciones=self._corpus_fuente.ruta_secciones.resolve(),
            ruta_xbrl=self._corpus_fuente.ruta_xbrl.resolve(),
            ruta_chunks=ruta_chunks,
            sha256_secciones=self._corpus_fuente.sha256_secciones,
            sha256_xbrl=self._corpus_fuente.sha256_xbrl,
            sha256_chunks=calcular_sha256(ruta_chunks),
            troceador=self._troceador.nombre,
            parametros_troceado=dict(self._troceador.parametros),
            artefactos_indice=artefactos,
            **campos_faiss,
        )

    @staticmethod
    def _remapear_variante(
        variante: CorpusVariant,
        *,
        origen: Path,
        destino: Path,
    ) -> CorpusVariant:
        def remapear(ruta: Path | None) -> Path | None:
            if ruta is None:
                return None
            relativa = ruta.resolve().relative_to(origen.resolve())
            return destino / relativa

        artefactos = variante.artefactos_indice
        artefactos_finales = None
        if artefactos is not None:
            indice = artefactos.indice.model_copy(
                update={"ruta": remapear(artefactos.indice.ruta)}
            )
            metadatos = (
                artefactos.metadatos.model_copy(
                    update={"ruta": remapear(artefactos.metadatos.ruta)}
                )
                if artefactos.metadatos is not None
                else None
            )
            artefactos_finales = artefactos.model_copy(
                update={"indice": indice, "metadatos": metadatos}
            )

        return variante.model_copy(
            update={
                "ruta_chunks": remapear(variante.ruta_chunks),
                "ruta_indice_faiss": remapear(variante.ruta_indice_faiss),
                "ruta_chunks_meta": remapear(variante.ruta_chunks_meta),
                "artefactos_indice": artefactos_finales,
            }
        )

    @staticmethod
    def _normalizar_nombre(nombre: str) -> str:
        if not isinstance(nombre, str):
            raise TypeError("nombre debe ser texto.")
        nombre = nombre.strip()
        if not nombre:
            raise ValueError("nombre no puede estar vacío.")
        return nombre
