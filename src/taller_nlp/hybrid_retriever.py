"""Recuperación híbrida mediante BM25, búsqueda densa y RRF."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from .chunking import FragmentoCorpus
from .corpus import CorpusVariant
from .retrieval import FragmentoRecuperado, Retriever


class _IndiceBM25(Protocol):
    def get_scores(self, query: list[str]) -> Any: ...


class RetrieverHibridoRRF(Retriever):
    """Fusiona por posiciones un ranking denso filtrado y otro BM25.

    Las puntuaciones de ambos recuperadores viven en escalas diferentes. RRF
    evita mezclarlas directamente y asigna a cada fragmento la suma de los
    recíprocos de sus posiciones en ambas listas.
    """

    _POSICION_AUSENTE = 10**6

    def __init__(
        self,
        corpus: CorpusVariant,
        *,
        retriever_denso: Retriever,
        indice_bm25: _IndiceBM25,
        fragmentos_bm25: Sequence[FragmentoCorpus | Mapping[str, object]],
        tokenizador: Callable[[str], list[str]],
        kk: int = 60,
        nombre: str = "hibrido-bm25-denso-rrf",
        nombre_tokenizador: str = "tokenizador",
    ) -> None:
        if not isinstance(retriever_denso, Retriever):
            raise TypeError(
                "retriever_denso debe ser una instancia de Retriever."
            )
        if retriever_denso.corpus != corpus:
            raise ValueError(
                "El retriever denso y el híbrido deben usar el mismo corpus."
            )
        if not retriever_denso.aplica_filtros_metadatos:
            raise ValueError(
                "El retriever denso debe aplicar filtros de metadatos."
            )
        if not callable(getattr(indice_bm25, "get_scores", None)):
            raise TypeError("indice_bm25 debe proporcionar get_scores().")
        if not callable(tokenizador):
            raise TypeError("tokenizador debe ser invocable.")
        if isinstance(kk, bool) or not isinstance(kk, int) or kk <= 0:
            raise ValueError("kk debe ser un entero mayor que cero.")

        fragmentos = tuple(
            fragmento
            if isinstance(fragmento, FragmentoCorpus)
            else FragmentoCorpus.model_validate(fragmento)
            for fragmento in fragmentos_bm25
        )
        if not fragmentos:
            raise ValueError("fragmentos_bm25 no puede estar vacío.")
        ids = tuple(fragmento.chunk_id for fragmento in fragmentos)
        if len(ids) != len(set(ids)):
            raise ValueError("fragmentos_bm25 contiene chunk_id duplicados.")

        super().__init__(
            nombre,
            corpus,
            parametros={
                "kk": kk,
                "posicion_ausente": self._POSICION_AUSENTE,
                "retriever_denso": retriever_denso.nombre,
                "tokenizador": nombre_tokenizador,
            },
            aplicar_filtros_metadatos=True,
        )
        self._retriever_denso = retriever_denso
        self._indice_bm25 = indice_bm25
        self._fragmentos_bm25 = fragmentos
        self._tokenizador = tokenizador
        self._kk = kk

    @property
    def retriever_denso(self) -> Retriever:
        return self._retriever_denso

    @property
    def kk(self) -> int:
        return self._kk

    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> Iterable[FragmentoRecuperado]:
        ranking_denso = self._retriever_denso.buscar(
            query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            item=item,
            k=len(self._fragmentos_bm25),
        )
        if not ranking_denso:
            return ()

        posiciones_densas = {
            fragmento.chunk_id: posicion
            for posicion, fragmento in enumerate(ranking_denso, start=1)
        }
        ids_admitidos = frozenset(posiciones_densas)

        puntuaciones_bm25 = self._indice_bm25.get_scores(
            self._tokenizador(query)
        )
        if len(puntuaciones_bm25) != len(self._fragmentos_bm25):
            raise ValueError(
                "BM25 devolvió un número de puntuaciones distinto al de "
                "fragmentos."
            )
        puntuaciones_lexicas = tuple(
            self._puntuacion_finita(puntuacion)
            for puntuacion in puntuaciones_bm25
        )
        orden_lexico = sorted(
            range(len(self._fragmentos_bm25)),
            key=lambda indice: (-puntuaciones_lexicas[indice], indice),
        )
        posiciones_lexicas = {
            self._fragmentos_bm25[indice].chunk_id: posicion
            for posicion, indice in enumerate(orden_lexico, start=1)
            if self._fragmentos_bm25[indice].chunk_id in ids_admitidos
        }
        scores_lexicos = {
            fragmento.chunk_id: puntuaciones_lexicas[indice]
            for indice, fragmento in enumerate(self._fragmentos_bm25)
            if fragmento.chunk_id in ids_admitidos
        }

        fusionados = []
        for fragmento in ranking_denso:
            posicion_densa = posiciones_densas[fragmento.chunk_id]
            posicion_lexica = posiciones_lexicas.get(
                fragmento.chunk_id, self._POSICION_AUSENTE
            )
            puntuacion_rrf = (
                1.0 / (self._kk + posicion_densa)
                + 1.0 / (self._kk + posicion_lexica)
            )
            detalles = dict(fragmento.detalles_puntuacion)
            detalles.update(
                {
                    "bm25": scores_lexicos.get(fragmento.chunk_id, 0.0),
                    "posicion_dense": float(posicion_densa),
                    "posicion_bm25": float(posicion_lexica),
                    "rrf": puntuacion_rrf,
                }
            )
            fusionados.append(
                FragmentoRecuperado(
                    **fragmento.model_dump(
                        exclude={"puntuacion", "detalles_puntuacion"}
                    ),
                    puntuacion=puntuacion_rrf,
                    detalles_puntuacion=detalles,
                )
            )

        fusionados.sort(
            key=lambda fragmento: (
                -float(fragmento.puntuacion),
                posiciones_densas[fragmento.chunk_id],
            )
        )
        return fusionados[:k]

    @staticmethod
    def _puntuacion_finita(valor: object) -> float:
        try:
            puntuacion = float(valor)
        except (TypeError, ValueError) as exc:
            raise TypeError("BM25 devolvió una puntuación no numérica.") from exc
        if not math.isfinite(puntuacion):
            raise ValueError("BM25 devolvió una puntuación no finita.")
        return puntuacion
