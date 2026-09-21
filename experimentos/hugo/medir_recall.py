"""Recall@k del retrieval aislado, sin LLM y sin coste.

Mide cada configuración de búsqueda sobre las preguntas con ancla de los dos
golden sets (el propio y el oficial). Es la medición que pide el enunciado
«después de cada arreglo, contra el ancla de texto».

Criterio de acierto: el de ``miax_s2.acierta`` (el que da el profesor para
que la métrica sea comparable entre grupos). Un fragmento cuenta si contiene
el ancla entera y es del mismo ticker y ejercicio: un factor de riesgo
copiado literal del año anterior no puntúa.

Condiciones de la medición, para que se lean bien los números:
- La consulta es la pregunta tal cual, en español, como en el notebook S2.
  El agente real escribe sus consultas en inglés, así que esto es una cota
  pesimista del recall que verá el agente.
- Los filtros salen de los metadatos del golden (ticker, ejercicio, item).
  En el agente los tiene que deducir el modelo de la pregunta.

Configuraciones, en el orden del notebook S2:
    1 denso plano · 2 + filtro de metadatos · 3 + híbrido BM25 (RRF)
    4 reescritura a inglés + filtro · 5 reescritura + híbrido
Las 4 y 5 llaman al LLM una vez por pregunta la PRIMERA vez; las
reescrituras se guardan en resultados/reescrituras_<modelo>.json y las
ejecuciones siguientes no cuestan nada. Sin clave de API se miden solo 1-3.

Uso, desde la raíz del repositorio:
    python -m experimentos.hugo.medir_recall
    python -m experimentos.hugo.medir_recall --k 10
    python -m experimentos.hugo.medir_recall --sin-reescritura
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

import miax_s2
from experimentos.baseline import RAIZ_PROYECTO, crear_corpus_baseline
from taller_nlp import RetrieverFaiss

GOLDEN_SETS = {
    "propio": RAIZ_PROYECTO / "golden_set.jsonl",
    "oficial": RAIZ_PROYECTO / "golden_set_oficial.jsonl",
}
DIRECTORIO_SALIDA = Path(__file__).resolve().parent / "resultados"

# Una búsqueda recibe el ítem del golden y un k, y devuelve fragmentos.
Busqueda = Callable[[dict, int], list[dict]]


def _a_dicts(fragmentos) -> list[dict]:
    return [fragmento.model_dump() for fragmento in fragmentos]


def crear_configuraciones(
    codificador: Any = None,
    reescribir: Callable[[str], str] | None = None,
) -> dict[str, Busqueda]:
    """Configuraciones a comparar, en el orden en que se aplican."""
    from experimentos.hugo.retrievers import RetrieverHibrido

    corpus = crear_corpus_baseline()
    plano = RetrieverFaiss(
        corpus,
        nombre="faiss-plano",
        codificador=codificador,
        aplicar_filtros_metadatos=False,
    )
    filtrado = RetrieverFaiss(
        corpus,
        nombre="faiss-filtro-metadatos",
        codificador=codificador,
        aplicar_filtros_metadatos=True,
    )

    def buscar_plano(g: dict, k: int) -> list[dict]:
        return _a_dicts(plano.buscar(g["pregunta"], k=k))

    def buscar_filtrado(g: dict, k: int) -> list[dict]:
        return _a_dicts(
            filtrado.buscar(
                g["pregunta"],
                ticker=g["ticker"],
                fiscal_year=g["fiscal_year"],
                item=g["item_esperado"],
                k=k,
            )
        )

    hibrido = RetrieverHibrido(filtrado)

    def con_filtros(retriever, consulta: str, g: dict, k: int) -> list[dict]:
        return _a_dicts(
            retriever.buscar(
                consulta,
                ticker=g["ticker"],
                fiscal_year=g["fiscal_year"],
                item=g["item_esperado"],
                k=k,
            )
        )

    configuraciones: dict[str, Busqueda] = {
        "1 · denso plano (baseline)": buscar_plano,
        "2 · + filtro de metadatos (v001)": buscar_filtrado,
        "3 · + híbrido BM25 (RRF)": lambda g, k: con_filtros(
            hibrido, g["pregunta"], g, k
        ),
    }
    if reescribir is not None:
        configuraciones["4 · reescritura + filtro"] = lambda g, k: con_filtros(
            filtrado, reescribir(g["pregunta"]), g, k
        )
        configuraciones["5 · reescritura + híbrido"] = lambda g, k: con_filtros(
            hibrido, reescribir(g["pregunta"]), g, k
        )
    return configuraciones


def cargar_con_ancla(ruta: Path) -> list[dict]:
    with ruta.open(encoding="utf-8") as fichero:
        golden = [json.loads(linea) for linea in fichero if linea.strip()]
    return [g for g in golden if g.get("ancla_texto")]


def medir(
    configuraciones: dict[str, Busqueda],
    k: int = 5,
    k_puesto: int = 1_749,
) -> pd.DataFrame:
    """Una fila por (golden set, configuración, pregunta)."""
    filas = []
    for nombre_golden, ruta in GOLDEN_SETS.items():
        for g in cargar_con_ancla(ruta):
            for nombre_config, buscar in configuraciones.items():
                ranking = buscar(g, k_puesto)
                filas.append(
                    {
                        "golden": nombre_golden,
                        "configuracion": nombre_config,
                        "id": g["id"],
                        "familia": g["familia"],
                        "item": g["item_esperado"],
                        f"acierta@{k}": miax_s2.acierta(g, ranking[:k]),
                        "puesto_ancla": miax_s2.posicion_del_ancla(g, ranking),
                    }
                )
    return pd.DataFrame(filas)


def resumir(tabla: pd.DataFrame, k: int) -> pd.DataFrame:
    columna = f"acierta@{k}"
    resumen = (
        tabla.groupby(["golden", "configuracion"])
        .agg(
            preguntas=("id", "size"),
            aciertos=(columna, "sum"),
            fallan=("id", lambda ids: ""),
        )
        .reset_index()
    )
    for indice, fila in resumen.iterrows():
        mascara = (
            (tabla.golden == fila.golden)
            & (tabla.configuracion == fila.configuracion)
            & (~tabla[columna])
        )
        resumen.at[indice, "fallan"] = ", ".join(tabla.loc[mascara, "id"])
    resumen[f"recall@{k}"] = (resumen.aciertos / resumen.preguntas).round(3)
    return resumen[
        ["golden", "configuracion", "preguntas", "aciertos", f"recall@{k}", "fallan"]
    ]


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--sin-reescritura",
        action="store_true",
        help="No medir las configuraciones que llaman al LLM.",
    )
    opciones = parser.parse_args(argumentos)

    reescritor = None
    if not opciones.sin_reescritura:
        from experimentos.hugo.comun import modelo_configurado
        from experimentos.hugo.retrievers import (
            ReescritorCacheado,
            crear_llamada_llm,
        )

        modelo = modelo_configurado()
        nombre_cache = "reescrituras_" + modelo.replace(":", "_").replace("/", "_")
        reescritor = ReescritorCacheado(
            crear_llamada_llm(modelo),
            DIRECTORIO_SALIDA / f"{nombre_cache}.json",
        )

    tabla = medir(crear_configuraciones(reescribir=reescritor), k=opciones.k)
    resumen = resumir(tabla, opciones.k)

    DIRECTORIO_SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DIRECTORIO_SALIDA / "recall_offline_detalle.csv", index=False)
    resumen.to_csv(DIRECTORIO_SALIDA / "recall_offline.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(resumen.to_string(index=False))
        print("\nPuesto del ancla (None = no aparece con esos filtros):")
        pivote = tabla.pivot_table(
            index=["golden", "id"],
            columns="configuracion",
            values="puesto_ancla",
            aggfunc="first",
        )
        print(pivote.to_string())
    if reescritor is not None:
        print(f"\nReescrituras nuevas con el LLM en esta ejecución: "
              f"{reescritor.llamadas_nuevas} (el resto, de la caché)")
    print(f"\nCSV en {DIRECTORIO_SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
