"""Recuperadores de Hugo: híbrido BM25 + denso y reescritura de consulta.

Los dos respetan el contrato ``Retriever`` del proyecto, así que sirven tanto
para medir el recall aislado (``medir_recall.py``) como para montarlos dentro
de ``search_filings`` en una versión del agente.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

import miax_s2
from taller_nlp import FragmentoRecuperado, Retriever, RetrieverFaiss


# Un único cerrojo para todo el proceso: la codificación con torch no es
# segura si dos hilos la ejecutan a la vez sobre el mismo modelo.
_CERROJO_CODIFICACION = threading.Lock()


class RetrieverFaissSeguro(RetrieverFaiss):
    """``RetrieverFaiss`` que codifica las consultas de una en una.

    Motivo (2026-09-20): cuando el modelo pide dos ``search_filings`` en el
    mismo turno, LangGraph ejecuta las dos herramientas en paralelo, cada una
    en un hilo. Los dos hilos entran a la vez en ``SentenceTransformer.encode``
    sobre el mismo modelo y el proceso muere con ``segmentation fault`` dentro
    de ``transformers/masking_utils.py``. Lo mostró ``faulthandler``: dos hilos
    en ``faiss_retriever._codificar`` en el momento del fallo. El cerrojo
    original de ``RetrieverFaiss`` solo protege la carga del modelo, no la
    codificación.

    No cambia el ranking: mismos vectores y mismos resultados, solo evita que
    dos codificaciones se solapen. Cada una tarda milisegundos.
    """

    def _codificar(self, query: str) -> Any:
        with _CERROJO_CODIFICACION:
            return super()._codificar(query)


class RetrieverHibrido(Retriever):
    """Fusiona el orden denso y el orden BM25 con Reciprocal Rank Fusion.

    RRF(d) = 1/(kk + puesto_denso(d)) + 1/(kk + puesto_bm25(d))

    Se combinan puestos y no puntuaciones porque el coseno y BM25 no están en
    la misma escala. Los filtros de metadatos los aplica el denso: BM25 solo
    ordena los fragmentos que ya han pasado el filtro. El tokenizador es el de
    ``miax_s2`` para que la medición sea comparable con la de clase.
    """

    def __init__(self, denso: RetrieverFaiss, *, kk: int = 60) -> None:
        if not denso.aplica_filtros_metadatos:
            raise ValueError("El denso del híbrido debe aplicar los filtros.")
        super().__init__(
            "hibrido-bm25-denso-rrf",
            denso.corpus,
            parametros={"kk": kk, "denso": denso.nombre},
            aplicar_filtros_metadatos=True,
        )
        self._denso = denso
        self._kk = kk
        with denso.corpus.ruta_chunks.open(encoding="utf-8") as fichero:
            textos = {}
            for linea in fichero:
                if linea.strip():
                    fila = json.loads(linea)
                    textos[fila["chunk_id"]] = fila["texto"]
        self._ids = list(textos)
        self._posicion = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        self._bm25 = BM25Okapi([miax_s2.tokenizar(t) for t in textos.values()])
        self._total = len(self._ids)

    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> Iterable[FragmentoRecuperado]:
        densos = self._denso.buscar(
            query, ticker=ticker, fiscal_year=fiscal_year, item=item, k=self._total
        )
        if not densos:
            return []
        puntuaciones_bm25 = self._bm25.get_scores(miax_s2.tokenizar(query))
        candidatos = [f.chunk_id for f in densos]
        orden_bm25 = sorted(
            candidatos,
            key=lambda cid: puntuaciones_bm25[self._posicion[cid]],
            reverse=True,
        )
        puesto_denso = {cid: i for i, cid in enumerate(candidatos, start=1)}
        puesto_bm25 = {cid: i for i, cid in enumerate(orden_bm25, start=1)}
        rrf = {
            cid: 1 / (self._kk + puesto_denso[cid]) + 1 / (self._kk + puesto_bm25[cid])
            for cid in candidatos
        }
        por_id = {f.chunk_id: f for f in densos}
        mejores = sorted(candidatos, key=rrf.__getitem__, reverse=True)[:k]
        return [
            por_id[cid].model_copy(
                update={
                    "puntuacion": rrf[cid],
                    "detalles_puntuacion": {
                        "rrf": rrf[cid],
                        "dense": float(por_id[cid].puntuacion or 0.0),
                        "bm25": float(puntuaciones_bm25[self._posicion[cid]]),
                        "puesto_denso": float(puesto_denso[cid]),
                        "puesto_bm25": float(puesto_bm25[cid]),
                    },
                }
            )
            for cid in mejores
        ]


INSTRUCCION_REESCRITURA = """Reescribe esta pregunta como una consulta de búsqueda para un
índice de informes 10-K en INGLÉS. Usa el vocabulario del propio informe.
Devuelve SOLO la consulta, sin comillas ni explicación."""
"""La misma instrucción que el notebook S2, para que el número sea comparable."""


class ReescritorCacheado:
    """Reescribe con un LLM y guarda el resultado en disco.

    La caché hace que la medición sea reproducible (misma consulta en cada
    ejecución) y que repetirla no cueste nada. La clave es la pregunta.
    """

    def __init__(self, llamar: Callable[[str], str], ruta_cache: Path) -> None:
        self._llamar = llamar
        self._ruta = ruta_cache
        self._cache: dict[str, str] = (
            json.loads(ruta_cache.read_text(encoding="utf-8"))
            if ruta_cache.is_file()
            else {}
        )
        self.llamadas_nuevas = 0

    def __call__(self, pregunta: str) -> str:
        if pregunta not in self._cache:
            self._cache[pregunta] = self._llamar(pregunta).strip()
            self.llamadas_nuevas += 1
            self._ruta.parent.mkdir(parents=True, exist_ok=True)
            self._ruta.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self._cache[pregunta]


def crear_llamada_llm(modelo: str) -> Callable[[str], str]:
    """Función pregunta -> consulta usando el modelo indicado."""
    from langchain.chat_models import init_chat_model

    from experimentos.hugo.comun import MODELOS_SIN_TEMPERATURA

    opciones = {"timeout": 60, "max_tokens": 200}
    if modelo not in MODELOS_SIN_TEMPERATURA:
        opciones["temperature"] = 0
    chat = init_chat_model(modelo, **opciones)

    def llamar(pregunta: str) -> str:
        return chat.invoke(
            [
                {"role": "system", "content": INSTRUCCION_REESCRITURA},
                {"role": "user", "content": pregunta},
            ]
        ).text

    return llamar


__all__ = [
    "INSTRUCCION_REESCRITURA",
    "ReescritorCacheado",
    "RetrieverFaissSeguro",
    "RetrieverHibrido",
    "crear_llamada_llm",
]
