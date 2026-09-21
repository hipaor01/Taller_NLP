"""Evalúa una variante y guarda el informe completo, no solo la tabla.

``python -m agente --evaluar`` guarda una fila por pregunta con True/False,
pero no qué contestó el agente. Sin eso no se puede saber POR QUÉ falla una
pregunta sin volver a pagarla. Este script guarda además el informe completo
(respuesta, cifra, unidad, cita, llamadas con sus argumentos) y muestra el
diagnóstico de las preguntas que fallan.

Salidas en ``experimentos/hugo/resultados/``:
    <etiqueta>.csv   la misma tabla que ``python -m agente`` (compatible con
                     ``python -m agente --resumir``)
    <etiqueta>.json  el informe completo, para diagnosticar sin reejecutar
    <etiqueta>_intervenciones.json  qué hicieron los middlewares propios
                     (v004 en adelante) y en qué pregunta

Uso, desde la raíz del repositorio:
    python -m experimentos.hugo.evaluar agente_v001 golden_set.jsonl v001_propio
    python -m experimentos.hugo.evaluar agente_v001 golden_set.jsonl v001_diag \
        --ids gjhh-007 gjhh-010

``--ids`` evalúa solo esas preguntas (útil para diagnosticar gastando poco).

Progreso reanudable: cada pregunta evaluada se guarda en
``resultados/progreso/<etiqueta>.json``. Si la ejecución se corta, volver a
lanzar EXACTAMENTE el mismo comando continúa por la primera pendiente sin
volver a pagar las anteriores. ``--reiniciar`` borra ese progreso y empieza
de cero. Si el proceso muere con ``segmentation fault``, ``faulthandler``
imprime en qué línea estaba.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from experimentos.baseline import RAIZ_PROYECTO

DIRECTORIO_RESULTADOS = Path(__file__).resolve().parent / "resultados"
FAMILIAS = ("extractiva", "numerica", "comparativa")


def _subconjunto(ruta: Path, ids: list[str]) -> Path:
    """Escribe un JSONL temporal con solo las preguntas pedidas."""
    filas = [
        linea
        for linea in ruta.read_text(encoding="utf-8").splitlines()
        if linea.strip() and json.loads(linea)["id"] in ids
    ]
    encontrados = {json.loads(linea)["id"] for linea in filas}
    faltan = sorted(set(ids) - encontrados)
    if faltan:
        raise SystemExit(f"No están en {ruta.name}: {', '.join(faltan)}")
    temporal = Path(tempfile.mkdtemp()) / ruta.name
    temporal.write_text("\n".join(filas) + "\n", encoding="utf-8")
    return temporal


def _resumen(informe) -> str:
    aciertos = informe.aciertos_por_familia
    totales = informe.preguntas_por_familia
    partes = [
        f"{familia} {aciertos[familia]}/{totales[familia]}"
        for familia in FAMILIAS
        if totales[familia]
    ]
    coste = informe.coste_medio_usd
    return (
        f"Aciertos: {' · '.join(partes)} · total "
        f"{informe.aciertos_totales}/{informe.numero_preguntas}\n"
        f"Coste medio {coste * 100:.2f} ¢ · latencia media "
        f"{informe.latencia_media_ms / 1000:.1f} s · "
        f"{informe.llamadas_por_pregunta:.2f} llamadas/pregunta"
        if coste is not None
        else f"Aciertos: {' · '.join(partes)}"
    )


def _diagnostico(informe, casos) -> str:
    por_id = {caso.id: caso for caso in casos}
    bloques = []
    for resultado in informe.resultados:
        if resultado.acierto:
            continue
        caso = por_id[resultado.id_pregunta]
        r = resultado.respuesta_agente
        llamadas = "\n".join(
            f"      {i}. {ll.nombre}({json.dumps(ll.argumentos, ensure_ascii=False)})"
            + (f"  ERROR: {ll.error[:120]}" if ll.error else "")
            for i, ll in enumerate(r.llamadas, 1)
        ) or "      (ninguna)"
        esperado = []
        if caso.cifra_esperada is not None:
            esperado.append(
                f"cifra={caso.cifra_esperada:,.0f} {caso.unidad} "
                f"concepto={caso.concept_xbrl}"
            )
        if caso.ancla_texto is not None:
            esperado.append(
                f"item={caso.item_esperado} ancla={caso.ancla_texto[:90]!r}"
            )
        cifra = f"{r.cifra:,.2f}" if r.cifra is not None else "None"
        bloques.append(
            f"\n--- {caso.id} ({caso.familia}) · "
            f"cita={resultado.cita_respalda} cifra={resultado.cifra_correcta} "
            f"trayectoria={resultado.trayectoria_correcta}\n"
            f"  pregunta : {caso.pregunta}\n"
            f"  esperado : {' | '.join(esperado)}\n"
            f"  respuesta: {r.respuesta[:300]}\n"
            f"  cifra    : {cifra} {r.unidad} · fuente={r.fuente}\n"
            f"  cita     : {(r.cita or 'None')[:160]!r} · chunk={r.citas}\n"
            f"  llamadas :\n{llamadas}"
            + (f"\n  error    : {r.error}" if r.error else "")
        )
    return "".join(bloques) or "\nSin fallos."


def _resumen_intervenciones(constructor, ruta: Path) -> str:
    """Vuelca qué hicieron los middlewares de la variante y lo resume."""
    from experimentos.hugo.guardrails import volcar_intervenciones

    if not volcar_intervenciones(constructor, ruta):
        return "\nMiddlewares: ninguna intervención registrada en esta tanda."
    intervenciones = json.loads(ruta.read_text(encoding="utf-8"))
    tipos = pd.Series([i["tipo"] for i in intervenciones]).value_counts()
    detalle = ", ".join(f"{tipo} {n}" for tipo, n in tipos.items())
    return f"\nMiddlewares: {detalle} (detalle en {ruta.name})"


def main(argumentos: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variante", help="Módulo dentro de experimentos.hugo, p. ej. agente_v001")
    parser.add_argument("golden", type=Path, help="JSONL a evaluar")
    parser.add_argument("etiqueta", help="Nombre base de los ficheros de salida")
    parser.add_argument("--ids", nargs="+", help="Evaluar solo estas preguntas")
    parser.add_argument(
        "--reiniciar",
        action="store_true",
        help="Descarta el progreso guardado de esta etiqueta y empieza de cero.",
    )
    opciones = parser.parse_args(argumentos)
    faulthandler.enable()

    ruta_progreso = DIRECTORIO_RESULTADOS / "progreso" / f"{opciones.etiqueta}.json"
    if opciones.reiniciar and ruta_progreso.exists():
        ruta_progreso.unlink()
    ruta_progreso.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HUGO_RUTA_PROGRESO"] = str(ruta_progreso)

    ruta = opciones.golden
    if not ruta.is_absolute():
        ruta = RAIZ_PROYECTO / ruta
    if opciones.ids:
        ruta = _subconjunto(ruta, opciones.ids)

    # La interfaz lee la variante de esta variable al construir el agente.
    os.environ["TALLER_VARIANTE_AGENTE"] = f"experimentos.hugo.{opciones.variante}"
    from agente import interfaz
    from taller_nlp import CasoGolden

    runtime = interfaz._obtener_runtime()
    informe = runtime.evaluar_informe(ruta)
    evaluador = runtime.constructor.evaluador
    casos = CasoGolden.cargar_jsonl(
        ruta,
        runtime.constructor.corpus,
        numero_esperado=evaluador.numero_esperado,
        minimo_comparativas=evaluador.minimo_comparativas,
    )
    tabla: pd.DataFrame = interfaz._informe_a_dataframe(informe, casos)

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_csv = DIRECTORIO_RESULTADOS / f"{opciones.etiqueta}.csv"
    ruta_json = DIRECTORIO_RESULTADOS / f"{opciones.etiqueta}.json"
    tabla.to_csv(ruta_csv, index=False)
    ruta_json.write_text(informe.model_dump_json(indent=2), encoding="utf-8")

    print(f"Variante: {runtime.constructor.nombre} · modelo "
          f"{runtime.constructor.configuracion.modelo}")
    print(_resumen(informe))
    print(_diagnostico(informe, casos))
    ruta_intervenciones = (
        DIRECTORIO_RESULTADOS / f"{opciones.etiqueta}_intervenciones.json"
    )
    print(_resumen_intervenciones(runtime.constructor, ruta_intervenciones))
    print(f"\nGuardado: {ruta_csv}\n          {ruta_json}")
    print(f"Progreso: {ruta_progreso} (bórralo con --reiniciar para repetir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
