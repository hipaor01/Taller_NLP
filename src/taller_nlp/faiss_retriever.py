"""Retriever denso respaldado por un índice FAISS."""

from __future__ import annotations

from threading import Lock
from typing import Any, Protocol

import faiss
import pandas as pd

from .corpus import CorpusVariant
from .retrieval import FragmentoRecuperado, Retriever


class _CodificadorEmbeddings(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> Any: ...


_COLUMNAS_REQUERIDAS = frozenset(
    {
        "chunk_id",
        "ticker",
        "fiscal_year",
        "item",
        "texto",
        "n_tokens",
        "contiene_tabla",
        "posicion",
        "inicio_car",
        "fin_car",
    }
)


class RetrieverFaiss(Retriever):
    """Búsqueda densa sobre los artefactos FAISS de ``CorpusVariant``.

    Puede ejecutar el ranking denso plano o aplicar después filtros de
    compañía, ejercicio e item. La opción queda registrada entre los
    parámetros del componente para que ambos experimentos sean distinguibles.
    """

    def __init__(
        self,
        corpus: CorpusVariant,
        *,
        nombre: str = "faiss",
        codificador: _CodificadorEmbeddings | None = None,
        normalizar_embeddings: bool | None = None,
        aplicar_filtros_metadatos: bool = True,
    ) -> None:
        if (
            corpus.ruta_indice_faiss is None
            or corpus.ruta_chunks_meta is None
            or corpus.modelo_embeddings is None
            or corpus.dimension_embeddings is None
        ):
            raise ValueError(
                "RetrieverFaiss requiere una CorpusVariant con índice FAISS, "
                "metadatos, modelo y dimensión de embeddings."
            )
        if codificador is not None and not callable(
            getattr(codificador, "encode", None)
        ):
            raise TypeError("codificador debe proporcionar un método encode().")
        normalizacion_indice = corpus.normalizar_embeddings
        if normalizar_embeddings is None:
            normalizar_embeddings = (
                normalizacion_indice
                if normalizacion_indice is not None
                else True
            )
        elif not isinstance(normalizar_embeddings, bool):
            raise TypeError("normalizar_embeddings debe ser bool o None.")
        elif (
            normalizacion_indice is not None
            and normalizar_embeddings != normalizacion_indice
        ):
            raise ValueError(
                "normalizar_embeddings no coincide con la configuración "
                "usada para construir el índice."
            )

        super().__init__(
            nombre,
            corpus,
            parametros={"normalizar_embeddings": normalizar_embeddings},
            aplicar_filtros_metadatos=aplicar_filtros_metadatos,
        )
        self._normalizar_embeddings = normalizar_embeddings
        self._codificador = codificador
        self._lock_codificador = Lock()
        self._indice = faiss.read_index(str(corpus.ruta_indice_faiss))
        self._metadatos = pd.read_parquet(corpus.ruta_chunks_meta)
        self._validar_metadatos()

    @property
    def normalizar_embeddings(self) -> bool:
        return self._normalizar_embeddings

    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> list[FragmentoRecuperado]:
        vector = self._codificar(query)
        hay_filtros = any(
            filtro is not None for filtro in (ticker, fiscal_year, item)
        )
        numero_candidatos = (
            self._indice.ntotal
            if self.aplica_filtros_metadatos and hay_filtros
            else min(k, self._indice.ntotal)
        )
        puntuaciones, posiciones = self._indice.search(
            vector, numero_candidatos
        )

        resultados = []
        for puntuacion, posicion_indice in zip(
            puntuaciones[0], posiciones[0]
        ):
            if posicion_indice < 0:
                continue
            fila = self._metadatos.iloc[int(posicion_indice)]
            if (
                self.aplica_filtros_metadatos
                and ticker is not None
                and fila["ticker"] != ticker
            ):
                continue
            if (
                self.aplica_filtros_metadatos
                and fiscal_year is not None
                and int(fila["fiscal_year"]) != fiscal_year
            ):
                continue
            if (
                self.aplica_filtros_metadatos
                and item is not None
                and fila["item"] != item
            ):
                continue

            resultados.append(
                self._crear_fragmento(fila, float(puntuacion))
            )
            if len(resultados) == k:
                break
        return resultados

    def _codificar(self, query: str) -> Any:
        consulta = self.corpus.prefijo_consulta + query
        vector = self._obtener_codificador().encode(
            [consulta],
            normalize_embeddings=self._normalizar_embeddings,
            convert_to_numpy=True,
        )
        if not hasattr(vector, "astype") or not hasattr(vector, "shape"):
            raise TypeError(
                "El codificador debe devolver un array con shape y astype()."
            )
        vector = vector.astype("float32")
        forma_esperada = (1, self._indice.d)
        if tuple(vector.shape) != forma_esperada:
            raise ValueError(
                f"El codificador devolvió shape={tuple(vector.shape)}; "
                f"se esperaba {forma_esperada}."
            )
        return vector

    def _obtener_codificador(self) -> _CodificadorEmbeddings:
        if self._codificador is None:
            with self._lock_codificador:
                if self._codificador is None:
                    from sentence_transformers import SentenceTransformer

                    assert self.corpus.modelo_embeddings is not None
                    self._codificador = SentenceTransformer(
                        self.corpus.modelo_embeddings
                    )
        assert self._codificador is not None
        return self._codificador

    def _validar_metadatos(self) -> None:
        columnas = frozenset(self._metadatos.columns)
        ausentes = _COLUMNAS_REQUERIDAS - columnas
        if ausentes:
            raise ValueError(
                f"Faltan columnas en los metadatos FAISS: {sorted(ausentes)}."
            )
        if self._metadatos[list(_COLUMNAS_REQUERIDAS)].isna().any().any():
            raise ValueError("Los metadatos FAISS contienen valores nulos.")

    @staticmethod
    def _crear_fragmento(
        fila: pd.Series, puntuacion: float
    ) -> FragmentoRecuperado:
        return FragmentoRecuperado(
            chunk_id=str(fila["chunk_id"]),
            ticker=str(fila["ticker"]),
            fiscal_year=int(fila["fiscal_year"]),
            item=str(fila["item"]),
            texto=str(fila["texto"]),
            n_tokens=int(fila["n_tokens"]),
            contiene_tabla=bool(fila["contiene_tabla"]),
            posicion=int(fila["posicion"]),
            inicio_car=int(fila["inicio_car"]),
            fin_car=int(fila["fin_car"]),
            puntuacion=puntuacion,
            detalles_puntuacion={"dense": puntuacion},
        )
