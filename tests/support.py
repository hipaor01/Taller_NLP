"""Fixtures pequeñas y dobles reutilizables, sin red ni modelos externos."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from taller_nlp import (
    CorteFragmento,
    CorpusVariant,
    FragmentoCorpus,
    FragmentoRecuperado,
    FabricaHerramientas,
    IndexadorCorpus,
    Retriever,
    TroceadorCorpus,
)
from taller_nlp.hashing import calcular_sha256


TEXTO_2024 = "Supply chain disruptions could materially harm operations."
TEXTO_2023 = "Demand changed because customers reduced their purchases."
ANCLA_2024 = "materially harm operations"


def crear_fragmento(
    texto: str = TEXTO_2024,
    *,
    ticker: str = "ACME",
    fiscal_year: int = 2024,
    item: str = "1A",
    posicion: int = 0,
    inicio_car: int = 0,
) -> FragmentoCorpus:
    return FragmentoCorpus(
        chunk_id=f"{ticker}-{fiscal_year}-{item}-{posicion:04d}",
        ticker=ticker,
        fiscal_year=fiscal_year,
        item=item,
        posicion=posicion,
        texto=texto,
        n_tokens=max(1, len(texto.split())),
        contiene_tabla=False,
        inicio_car=inicio_car,
        fin_car=inicio_car + len(texto),
    )


def como_recuperado(
    fragmento: FragmentoCorpus, puntuacion: float = 1.0
) -> FragmentoRecuperado:
    return FragmentoRecuperado(
        **fragmento.model_dump(),
        puntuacion=puntuacion,
        detalles_puntuacion={"test": puntuacion},
    )


def escribir_jsonl(ruta: Path, filas: Iterable[dict]) -> None:
    contenido = "".join(
        json.dumps(fila, ensure_ascii=False) + "\n" for fila in filas
    )
    ruta.write_text(contenido, encoding="utf-8")


def crear_corpus_temporal(
    raiz: Path,
    *,
    nombre: str = "fixture",
    con_faiss: bool = False,
) -> tuple[CorpusVariant, tuple[FragmentoCorpus, ...]]:
    raiz.mkdir(parents=True, exist_ok=True)
    ruta_secciones = raiz / "secciones.jsonl"
    ruta_xbrl = raiz / "xbrl_facts.parquet"
    ruta_chunks = raiz / "chunks.jsonl"

    secciones = [
        {
            "cik": 1,
            "ticker": "ACME",
            "empresa": "Acme Corporation",
            "fiscal_year": 2024,
            "item": "1A",
            "texto": TEXTO_2024,
        },
        {
            "cik": 1,
            "ticker": "ACME",
            "empresa": "Acme Corporation",
            "fiscal_year": 2023,
            "item": "7",
            "texto": TEXTO_2023,
        },
    ]
    escribir_jsonl(ruta_secciones, secciones)
    pd.DataFrame(
        [
            {
                "ticker": "ACME",
                "fiscal_year": 2024,
                "concept": "NetIncomeLoss",
                "value": 100.0,
                "unit": "USD",
                "period_end": "2024-12-31",
                "form": "10-K",
            },
            {
                "ticker": "ACME",
                "fiscal_year": 2023,
                "concept": "NetIncomeLoss",
                "value": 80.0,
                "unit": "USD",
                "period_end": "2023-12-31",
                "form": "10-K",
            },
        ]
    ).to_parquet(ruta_xbrl, index=False)

    fragmentos = (
        crear_fragmento(),
        crear_fragmento(
            TEXTO_2023,
            fiscal_year=2023,
            item="7",
        ),
    )
    escribir_jsonl(ruta_chunks, (f.model_dump(mode="json") for f in fragmentos))

    campos_faiss: dict[str, object] = {}
    if con_faiss:
        directorio_indice = raiz / "indice"
        directorio_indice.mkdir()
        ruta_indice = directorio_indice / "corpus.faiss"
        ruta_meta = directorio_indice / "chunks_meta.parquet"
        indice = faiss.IndexFlatIP(2)
        indice.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
        faiss.write_index(indice, str(ruta_indice))
        pd.DataFrame([f.model_dump() for f in fragmentos]).to_parquet(
            ruta_meta, index=False
        )
        campos_faiss = {
            "ruta_indice_faiss": ruta_indice,
            "ruta_chunks_meta": ruta_meta,
            "modelo_embeddings": "modelo-de-prueba",
            "dimension_embeddings": 2,
            "prefijo_consulta": "query: ",
        }

    corpus = CorpusVariant(
        nombre=nombre,
        ruta_secciones=ruta_secciones,
        ruta_xbrl=ruta_xbrl,
        ruta_chunks=ruta_chunks,
        sha256_secciones=calcular_sha256(ruta_secciones),
        sha256_xbrl=calcular_sha256(ruta_xbrl),
        sha256_chunks=calcular_sha256(ruta_chunks),
        troceador="fixture",
        **campos_faiss,
    )
    return corpus, fragmentos


def crear_fabrica_prueba(corpus: CorpusVariant) -> FabricaHerramientas:
    def buscar(
        corpus: CorpusVariant,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> str:
        del corpus, query, ticker, fiscal_year, item, k
        return "sin resultados"

    return FabricaHerramientas(corpus, search_filings_impl=buscar)


class TroceadorSeccionCompleta(TroceadorCorpus):
    def __init__(self) -> None:
        super().__init__("seccion-completa", parametros={"unidad": "seccion"})

    def _calcular_cortes(
        self,
        *,
        texto: str,
        ticker: str,
        fiscal_year: int,
        item: str,
    ) -> Iterable[CorteFragmento]:
        del ticker, fiscal_year, item
        return (CorteFragmento(
            inicio_car=0,
            fin_car=len(texto),
            n_tokens=len(texto.split()),
        ),)


class RetrieverControlado(Retriever):
    def __init__(
        self,
        corpus: CorpusVariant,
        resultados: Sequence[FragmentoRecuperado],
    ) -> None:
        super().__init__("controlado", corpus)
        self.resultados = tuple(resultados)
        self.ultima_entrada: dict[str, object] | None = None

    def _buscar(
        self,
        *,
        query: str,
        ticker: str | None,
        fiscal_year: int | None,
        item: str | None,
        k: int,
    ) -> Iterable[FragmentoRecuperado]:
        self.ultima_entrada = {
            "query": query,
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "item": item,
            "k": k,
        }
        return self.resultados


class IndexadorTexto(IndexadorCorpus):
    def __init__(self) -> None:
        super().__init__(
            "indice-texto",
            "test",
            parametros={"version": 1},
        )
        self.fragmentos_recibidos: tuple[FragmentoCorpus, ...] = ()

    def _construir_indice(
        self,
        fragmentos: tuple[FragmentoCorpus, ...],
        directorio_salida: Path,
    ) -> tuple[Path, Path | None]:
        self.fragmentos_recibidos = fragmentos
        indice = directorio_salida / "indice.txt"
        metadatos = directorio_salida / "meta.json"
        indice.write_text("\n".join(f.texto for f in fragmentos), encoding="utf-8")
        metadatos.write_text(
            json.dumps([f.chunk_id for f in fragmentos]), encoding="utf-8"
        )
        return indice, metadatos


class CodificadorControlado:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = np.asarray([vector], dtype="float32")
        self.llamadas: list[tuple[list[str], bool, bool]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.llamadas.append(
            (sentences, normalize_embeddings, convert_to_numpy)
        )
        return self.vector
